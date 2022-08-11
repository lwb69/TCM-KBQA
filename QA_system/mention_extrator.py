import jieba
import codecs as cs
import time
import torch
from qa.system.QA_system.train_ner import Bert_NER
from transformers import BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MentionExtractor(object):
    def __init__(self, ):
        # 分词词典加载
        begin = time.time()
        with cs.open('qa/system/data/segment_dic.txt', 'r', 'utf-8') as fp:
            segment_dic = {}
            for line in fp:
                if line.strip():
                    segment_dic[line.strip()] = 0
        jieba.load_userdict('qa/system/data/segment_dic.txt')
        self.segment_dic = segment_dic
        self.max_seq_len = 20
        print('加载用户分词词典时间为:%.2f' % (time.time() - begin))
        # 加载训练好的实体识别模型
        self.model = Bert_NER()
        self.model.to(device)
        checkpoint = torch.load('qa/system/models/ner_model-best_1.pkl')
        self.model.load_state_dict(checkpoint['Bert_NER_state_dict'])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='qa/system/pretrainmodels/rbt3')
        print('mention extractor loaded')

    def restore_entity_from_labels_on_corpus(self, pred_labels, question):
        entitys = []
        str = ''
        labels = pred_labels[1:-1]
        for i in range(min(len(labels), len(question))):
            if labels[i] == 1:
                str += question[i]
            else:
                if len(str):
                    entitys.append(str)
                    str = ''
        if len(str):
            entitys.append(str)
        return entitys

    def extract_mentions(self, question):
        '''
                返回mention字典
        '''
        entity_mention = {}
        mentions = []
        tokens = jieba.lcut(question)
        features = []
        for t in tokens:
            if t in self.segment_dic:
                mentions.append(t)
        features.append(self.tokenizer.encode(text=question, max_length=self.max_seq_len))

        features = torch.tensor(features, dtype=torch.long)
        self.model.eval()
        features = features.to(device)
        with torch.no_grad():
            predict = self.model(features)
        pred_labels = [1 if each[0] > 0.5 else 0 for each in predict[0]]
        pred_entity = self.restore_entity_from_labels_on_corpus(pred_labels, question)
        me = mentions + pred_entity
        me =set(me)

        for token in me:
            entity_mention[token]=token

        max=0
        for mention in entity_mention.keys():
            overlap = len(set(mention)&(set(question)))
            if overlap>max:
                filter_mention=mention
                max=overlap
        filter_entity_mention={}
        filter_entity_mention[filter_mention]=entity_mention[filter_mention]
        return filter_entity_mention


if __name__ == '__main__':
    a = MentionExtractor()
    ans = a.extract_mentions('邪郁少阳证的判断属于什么辩证法')
