import torch
from sys import platform
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data import Dataset
from hanziconv import HanziConv
import pandas as pd
from transformers import BertForSequenceClassification, BertConfig
import time
from sklearn.metrics import roc_auc_score
class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, max_char_len=32):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = BertTokenizer.from_pretrained('qa/system/pretrainmodels/Robert_for_similarity/vocabs.txt', do_lower_case=True)
        self.max_seq_len = max_char_len
        # self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(data_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]


    # 获取文本与标签
    def get_input(self,data_list):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """
        sentences_1=[item[0] for item in data_list]
        sentences_2=[item[1] for item in data_list]
        # labels = df['label'].values
        # 切词
        tokens_seq_1 = list(map(self.bert_tokenizer.tokenize, sentences_1))
        tokens_seq_2 = list(map(self.bert_tokenizer.tokenize, sentences_2))
        # 获取定长序列及其mask
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(
            seq_segments).type(torch.long)

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        入参:
            seq_1       : 输入序列，在本处其为单个句子。
            seq_2       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为0，否则为1。
            seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分

        """
        # 对超长序列进行截断
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3) // 2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3) // 2]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment

class BertModel_Simmer(nn.Module):
    def __init__(self):
        super(BertModel_Simmer, self).__init__()
        config = BertConfig.from_pretrained('qa/system/pretrainmodels/Robert_for_similarity')
        self.bert = BertForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments):
        output = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks,token_type_ids=batch_seq_segments)
        probabilities = nn.functional.softmax(output.logits, dim=-1)
        return probabilities

def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:,1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(all_labels, all_prob)
def main(test_file, pretrained_file, batch_size=32):

    device = torch.device("cuda")
    bert_tokenizer = BertTokenizer.from_pretrained('../pretrainmodels/Robert_for_similarity/vocabs.txt', do_lower_case=True)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(bert_tokenizer, test_file)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    model = BertModel_Simmer().to(device)
    model.load_state_dict(checkpoint["model"],False)
    print(20 * "=", " Testing roberta model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))


if __name__ == "__main__":
    main("models/lcqmc/test.tsv", "models/models/best.pth.tar")