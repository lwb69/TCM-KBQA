import pickle
import random
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.nn.functional import binary_cross_entropy
import torch.optim as optim
from tqdm import tqdm,trange
import qa.system.QA_system.utils
import numpy as np
import csv
import argparse
import jieba
import codecs as cs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_corpus(state):
    '''
    加载并划分数据集
    :param state: 'train',''valid','test'  path:''str
    :return: question[],entity[]
    '''

    if state == 'train':
        corpus=pickle.load(open('../data/train_corpus.pkl','rb'))
        train_questions = [corpus[i]['question'] for i in corpus.keys()]
        train_entity = [corpus[i]['gold_entitys'] for i in corpus.keys()]
        return train_questions, train_entity
    elif state == 'valid':
        corpus = pickle.load(open('../data/valid_corpus.pkl', 'rb'))
        valid_questions = [corpus[i]['question'] for i in corpus.keys()]
        valid_entity = [corpus[i]['gold_entitys'] for i in corpus.keys()]
        return valid_questions, valid_entity
    # elif state == 'test':
    #     test_questions = [corpus[i]['question'] for i in test_index]
    #     test_entity = [corpus[i]['gold_entitys'] for i in test_index]
    #     return test_questions, test_entity


def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
            if m[i + 1][j + 1] > mmax:
                mmax = m[i + 1][j + 1]
                p = i + 1
    return s1[p - mmax:p]


class Bert_NER(nn.Module):
    def __init__(self):
        super(Bert_NER, self).__init__()
        # pretrainmodel
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='qa/system/pretrainmodels/rbt3')
        # 使参数可更新
        for param in self.bert.parameters():
            param.requires_grad = True
        self.ds_encoder = nn.LSTM(768, 256, bidirectional=True)
        self.fc = nn.Linear(512, 1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        outputs = self.bert(x,output_hidden_states=True)
        features=outputs.last_hidden_state
        state = self.ds_encoder(features)
        preds = self.sigmoid((self.fc(state[0])))

        return preds


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class NERDataLoader(object):
    def __init__(self):
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.test_batch_size = 16
        self.data_cache = True
        self.max_seq_len = 25
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='../pretrainmodels/rbt3')

    @staticmethod
    def collate_fn(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
        labels = torch.tensor([f[1]for f in features], dtype=torch.long)
        tensors = [input_ids, labels]
        return tensors

    def convert_examples_to_features(self, questions, entitys):
        #除了用问句和对应实体，需要用提及词典来反向标注问句
        entity2mention_dic = pickle.load(open('../data/entity2mention_dic_cm3.pkl', 'rb'))
        features = []
        for i in range(len(questions)):
            q = questions[i]
            x = self.tokenizer.encode(text=q, max_length=self.max_seq_len)
            y = [[0] for j in range(self.max_seq_len)]
            # padding
            if len(x) != len(y):
                x.extend([0] * (len(y) - len(x)))
            assert len(x) == len(y)
            for e in entitys[i]:
                # 得到实体名和问题的最长连续公共子串
                e1 = find_lcsubstr(e, q)
                if e1 in q and e1!='':
                    begin = q.index(e1) + 1
                    end = begin + len(e1)
                    if end < self.max_seq_len - 1:
                        for pos in range(begin, end):
                            y[pos] = [1]
                else:
                    for enty,mentions in entity2mention_dic.items():
                        if e in enty:
                            for mention in mentions:
                                e2=find_lcsubstr(mention, q)
                                if e2 in q and e2!='':
                                    begin = q.index(e2) + 1
                                    end = begin + len(e2)
                                    if end < self.max_seq_len - 1:
                                        for pos in range(begin, end):
                                            y[pos] = [1]
            print(q)
            print(q[begin-1:end-1])
            # print(x)
            # print(y)
            if not [1] in y:
                print(q,e)
            features.append([x, y])
        return features
    def get_dataloaderbypath(self,path):
        corpus=pickle.load(open(path,'rb'))
        questions =[corpus[i]['question'] for i in corpus.keys()]
        entities = [corpus[i]['gold_entitys'] for i in corpus.keys()]
        features = self.convert_examples_to_features(questions, entities)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {path} data loaded!")
        print("=*=" * 10)
        datasampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                collate_fn=self.collate_fn,shuffle=False)
        return dataloader
    def get_dataloader(self, data_sign):
        questions, entity = load_corpus(state=data_sign)
        features = self.convert_examples_to_features(questions, entity)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)

        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn)
        elif data_sign == "valid":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn)
        elif data_sign in ("test", "pseudo"):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn)
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader


def train(epoches,train_loader,val_loader):
    # dataloader = NERDataLoader()
    # train_loader= dataloader.get_dataloader(data_sign='train')
    # print(len(train_loader))
    # val_loader= dataloader.get_dataloader(data_sign='valid')
    model = Bert_NER()
    model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=1e-5)
    maxf=0
    for epoch in range(1, epoches + 1):
        optimizer.zero_grad()
        print("Epoch {}/{}".format(epoch, epoches))
        model.train()
        loss_avg=utils.RunningAverage()
        t = trange(len(train_loader), ascii=True)
        for step, _ in enumerate(t):
            # fetch the next training batch
            batch = next(iter(train_loader))
            batch = tuple(t.to(device) for t in batch)
            input_ids,labels=batch

            predicts=model(input_ids)

            loss=binary_cross_entropy(predicts,labels.float())

            loss.backward()
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss.item()),avg_loss='{:05.3f}'.format(loss_avg()))
            optimizer.step()

        #evaluate
        _,p,r,f=evaluate(model,val_loader)
        print('{}epoch p is{:.3f} R is {:.3f} f-score is {:.3f}'.format(epoch, p, r, f))
        #模型保存
        if f > maxf:
            torch.save({
                'Bert_NER_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },'../models/ner_model-best_1.pkl')
            with open('../models/ner_model-best_1.csv', 'a+') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'loss_avg', 'precision', 'recall', 'f1'])
                writer.writerow([epoch,loss_avg(),p,r,f])
            maxf = f

def evaluate(model,data_loader):
    model.eval()
    predicts=[]
    labels_all=[]
    for batch in tqdm(data_loader, unit='Batch', ascii=True):
        batch = tuple(t.to(device) for t in batch)
        input_ids,labels=batch
        batch_size, max_len,_ = labels.size()
        with torch.no_grad():
            predict = model(input_ids)
            predicts.extend(predict)
        labels=labels.to('cpu'). numpy()
        labels=labels.squeeze().tolist()
        labels_all.extend(labels)
    pred_labels = [[1 if each[0] > 0.5 else 0 for each in line] for line in predicts ]
    val_ques,val_entity=load_corpus('valid')
    pred_entity=restore_entity_from_labels_on_corpus(pred_labels, val_ques)
    true_entity=restore_entity_from_labels_on_corpus(labels_all,val_ques)
    p, r, f = computeF( true_entity,pred_entity)
    return pred_entity,p,r,f
def restore_entity_from_labels_on_corpus(predicty,questions):
    def restore_entity_from_labels(labels,question):
        entitys = []
        str = ''
        labels = labels[1:-1]
        for i in range(min(len(labels),len(question))):
            if labels[i]==1:
                str += question[i]
            else:
                if len(str):
                    entitys.append(str)
                    str = ''
        if len(str):
            entitys.append(str)
        return entitys
    all_entitys = []
    for i in range(len(predicty)):
        all_entitys.append(restore_entity_from_labels(predicty[i],questions[i]))
    return all_entitys

def computeF(gold_entity,pre_entity):
    '''
        根据标注的实体位置和预测的实体位置，计算prf,完全匹配
        输入： Python-list  3D，值为每个实体的起始位置列表[begin，end]
        输出： float
        '''
    truenum = 0
    prenum = 0
    goldnum = 0
    for i in range(len(gold_entity)):
        goldnum += len(gold_entity[i])
        prenum += len(pre_entity[i])
        truenum += len(set(gold_entity[i]).intersection(set(pre_entity[i])))
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall / (precise + recall))
    except:
        precise = recall = f = 0.0
    return precise, recall, f
def mention_extrate(corpus,model,data_loader):
    model.eval()
    predicts = []
    labels_all = []
    for batch in tqdm(data_loader, unit='Batch', ascii=True):
        batch = tuple(t.to(device) for t in batch)
        input_ids, labels = batch
        batch_size, max_len, _ = labels.size()
        with torch.no_grad():
            predict = model(input_ids)
            predicts.extend(predict)
        labels = labels.to('cpu').numpy()
        labels = labels.squeeze().tolist()
        labels_all.extend(labels)

    ques=[corpus[i]['question'] for i in range(len(corpus))]
    pred_labels = [[1 if each[0] > 0.5 else 0 for each in line] for line in predicts]
    pred_entity = restore_entity_from_labels_on_corpus(pred_labels, ques)

    return pred_entity
def getmentionfordevlop(corpus,path):
    #利用bert模型进行mention抽取
    model = Bert_NER()
    model.to(device)
    checkpoint = torch.load('../models/ner_model-best_1.pkl')
    model.load_state_dict(checkpoint['Bert_NER_state_dict'])
    dataloader = NERDataLoader()
    data_loader = dataloader.get_dataloaderbypath(path=path)
    pred_entity=mention_extrate(corpus,model,data_loader)
    # 分词词典分词提取mention
    with cs.open('../data/segment_dic.txt', 'r', 'utf-8') as fp:
        segment_dic = {}
        for line in fp:
            if line.strip():
                segment_dic[line.strip()] = 0
    jieba.load_userdict('../data/segment_dic.txt')

    for i in range(len(corpus)):
        dic = corpus[i]
        question = dic['question']
        entity_mention={}
        mentions = []
        tokens = jieba.lcut(question)
        for t in tokens:
            if t in segment_dic:
                mentions.append(t)
        me= mentions + pred_entity[i]
        for token in me:
            entity_mention[token] = token
        dic['entity_mention'] = entity_mention
        corpus[i] = dic
        print(question)
        print(dic['entity_mention'])
    print("问句数量：",len(corpus))

    return corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train or use the NER model')
    parser.add_argument('--trainortest', type=str,default='train')
    args=parser.parse_args()
    setup_seed(20)
    #加载数据集
    if args.trainortest=='train':
        dataloader = NERDataLoader()
        train_loader = dataloader.get_dataloader(data_sign='train')
        val_loader = dataloader.get_dataloader(data_sign='valid')
        train(epoches=30,train_loader=train_loader,val_loader=val_loader)
    elif args.trainortest=='test':
        inputpaths=['../data/train_corpus.pkl', '../data/valid_corpus.pkl']
        outputpaths=['../data/entity_mentions_train.pkl','../data/entity_mentions_valid.pkl']
        for i in range(len(inputpaths)):
            inputpath=inputpaths[i]
            outputpath=outputpaths[i]
            corpus=pickle.load(open(inputpath,'rb'))
            corpus=getmentionfordevlop(corpus=corpus,path=inputpath)
            pickle.dump(corpus, open(outputpath, 'wb'))
            print('得到实体mention')



