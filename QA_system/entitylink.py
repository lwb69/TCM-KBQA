import pickle
from qa.system.QA_system.utils import ComputeEntityFeatures
from qa.system.QA_system.KG import KGapi
import codecs as cs
import time

errors=[]
class EntityExtractor(object):
    def __init__(self):
        self.mention2entity_dic = pickle.load(open('qa/system//data/mention2entity_dic_cm4.pkl', 'rb'))
        allseq_dict = {}
        with open('qa/system/KG/allseg.txt', 'r', encoding='utf-8') as fp:
            for line in fp:
                if line:
                    ename = line.strip('\n')
                    allseq_dict[ename] = 1
        self.allseq_dict=allseq_dict
        print('mention in self.allseq_dict',('麻杏汤' in self.allseq_dict))
        self.KGapi=KGapi()
        try:
            self.entity2hop_dic = pickle.load(open('qa/system/data/entity2hop_dic.pkl', 'rb'))
        except:
            self.entity2hop_dic = {}
        self.fp = cs.open('qa/system/data/record/entity_extractor_ans.txt', 'w')
        print('entity extractor loaded')

    def get_mention_feature(self, question, mention):
        f1 = float(len(mention))  # mention的长度

        try:
            f2 = float(question.index(mention))
        except:
            f2 = 3.0
            # print ('这个mention无法提取位置')
        return [mention, f1, f2]

    def extract_subject(self, entity_mentions, question):
        '''
        根据前两部抽取出的实体mention，得到实体
         Input:
                entity_mentions: {str:list} {'脱发':'脱发'}
        output:
                    candidate_subject: {str:list}
        '''

        candidate_entity = {}
        print('开始遍历')
        for mention in entity_mentions.keys():  # 遍历每一个mention
            print('====当前实体mention为：%s====' % (mention))
            if mention in self.mention2entity_dic:  # 如果它有对应的实体
                print("mention在链接词典中")
                for entity in self.mention2entity_dic[mention]:
                    # mention的特征
                    mention_features = self.get_mention_feature(question, mention)  # 包括mention的长度，mention在问句中的起始位置
                    # 得到实体一跳内的所有关系和属性
                    if entity in self.entity2hop_dic:
                        relations = self.entity2hop_dic[entity]
                    else:
                        relations = self.KGapi.GetRelationPathsSingle(entity)
                        self.entity2hop_dic[entity] = relations
                    # 计算问题和主语实体及其一跳内关系间的相似度
                    nametypes=[]
                    relsdic_list=[]
                    for nametype, rel in relations.items():
                        nametypes.append(nametype)
                        relsdic_list.append(rel)
                    if len(relsdic_list)==1:
                        relations_list=[]
                        for r in relsdic_list[0].values():
                            relations_list.extend(r)
                        similar_features = ComputeEntityFeatures(question, entity, relations_list)
                        # 实体的流行度特征
                        popular_feature = self.KGapi.GetRelationNum(entity)

                        candidate_entity[entity] = mention_features + similar_features + [popular_feature ** 0.5]
                    else:
                        errors.append([question,mention])
            elif mention in self.allseq_dict:#可能在all_seq中,即mention即为entity
                print("mention在实体词典中")
                # mention的特征
                print('mention值为:',mention)
                entity=mention
                mention_features = self.get_mention_feature(question, mention)  # 包括mention的长度，mention在问句中的起始位置
                # 得到实体一跳内的所有关系和属性
                if entity in self.entity2hop_dic:
                    relations = self.entity2hop_dic[mention]
                else:
                    relations = self.KGapi.GetRelationPathsSingle(entity)
                    self.entity2hop_dic[entity] = relations
                # 计算问题和主语实体及其一跳内关系间的相似度
                nametypes = []
                relsdic_list = []
                for nametype, rel in relations.items():
                    nametypes.append(nametype)
                    relsdic_list.append(rel)
                if len(relsdic_list) == 1:
                    relations_list = []
                    for r in relsdic_list[0].values():
                        relations_list.extend(r)
                    similar_features = ComputeEntityFeatures(question, entity, relations_list)
                    # 实体的流行度特征
                    popular_feature = self.KGapi.GetRelationNum(entity)

                    candidate_entity[entity] = mention_features + similar_features + [popular_feature ** 0.5]
                else:
                    errors.append([question, mention])
            else:
                print("mention需要字符串相似度链接")
                max_jaccard=0
                for seq in self.allseq_dict:
                    jaccard = len(set(mention)&(set(seq))) / len(set(mention)|(set(seq)))
                    if jaccard > max_jaccard:
                        entity = seq
                        max_jaccard=jaccard
                mention_features = self.get_mention_feature(question, mention)  # 包括mention的长度，mention在问句中的起始位置
                # 得到实体一跳内的所有关系和属性
                if entity in self.entity2hop_dic:
                    relations = self.entity2hop_dic[entity]
                else:
                    relations = self.KGapi.GetRelationPathsSingle(entity)
                    self.entity2hop_dic[entity] = relations
                # 计算问题和主语实体及其一跳内关系间的相似度
                nametypes = []
                relsdic_list = []
                for nametype, rel in relations.items():
                    nametypes.append(nametype)
                    relsdic_list.append(rel)
                if len(relsdic_list) == 1:
                    relations_list = []
                    for r in relsdic_list[0].values():
                        relations_list.extend(r)
                    similar_features = ComputeEntityFeatures(question, entity, relations_list)
                    # 实体的流行度特征
                    popular_feature = self.KGapi.GetRelationNum(entity)

                    candidate_entity[entity] = mention_features + similar_features + [popular_feature ** 0.5]
                else:
                    errors.append([question, mention])

        if len(candidate_entity)==0:
            print(entity_mentions, question)
        pickle.dump(self.entity2hop_dic, open('qa/system/data/entity2hop_dic.pkl', 'wb'))
        return candidate_entity

    def GetCandidateEntity(self, corpus):
        true_num = 0.0
        one_num = 0.0
        one_true_num = 0.0
        subject_num = 0.0
        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            gold_entitys = dic['gold_entitys']
            # candidate_entity = {}
            # print('\n')
            # print(i)
            # print(question)
            starttime = time.time()
            # 得到当前问题的候选主语mention
            entity_mentions = dic['entity_mention']
            candidate_entity = self.extract_subject(entity_mentions, question)
            subject_num += len(candidate_entity)
            dic['candidate_entity'] = candidate_entity
            # print('候选实体为：')
            for c in candidate_entity:
                print(c, candidate_entity[c])
            print('candidate_entity长度',len(candidate_entity))
            print('耗费时间%.2f秒' % (time.time() - starttime))

            if len(set(gold_entitys)) == len(set(gold_entitys).intersection(set(candidate_entity))):
                true_num += 1
                if len(gold_entitys) == 1:
                    one_true_num += 1
            else:
                print(question)
                print(gold_entitys)
                print(candidate_entity.keys)
                self.fp.write(str(i) + question + '\n')
                self.fp.write('\t'.join(gold_entitys) + '\n')
                self.fp.write('\t'.join(list(candidate_entity.keys())) + '\n\n')
                self.fp.write('\t'.join(entity_mentions) + '\n\n')
            if len(gold_entitys) == 1:
                one_num += 1

        pickle.dump(self.entity2hop_dic, open('../data/entity2hop_dic.pkl', 'wb'))
        print('单实体问题可召回比例为%.2f' % (one_true_num / one_num))
        print('所有问题可召回比例为%.2f' % (true_num / len(corpus)))
        print('平均每个问题的候选主语个数为:%.2f' % (subject_num / len(corpus)))
        print(errors)
        return corpus


if __name__ == '__main__':
    inputpaths = ['../data/entity_mentions_train.pkl', '../data/entity_mentions_valid.pkl']
    # inputpaths = [ '../data/entity_mentions_valid.pkl']
    outputpaths = ['../data/candidate_entitys_train.pkl', '../data/candidate_entitys_valid.pkl']
    ee = EntityExtractor()
    for i in range(len(inputpaths)):
        inputpath = inputpaths[i]
        outputpath = outputpaths[i]
        corpus = pickle.load(open(inputpath, 'rb'))
        corpus = ee.GetCandidateEntity(corpus)
        pickle.dump(corpus, open(outputpath, 'wb'))
