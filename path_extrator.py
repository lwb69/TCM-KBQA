import pickle
import time
from KG import GetRelationPaths,GetRelationPathsSingle
from similarity_bert import BertModel_Simmer,DataPrecessForSentence
import torch
import codecs as cs
import random
random.seed(30)
class PathExtractor(object):
    def __init__(self):

        # 加载一些缓存
        try:
            self.entity2relations_dic = pickle.load(open('../data/entity2relation_dic.pkl', 'rb'))
        except:
            self.entity2relations_dic = {}
        self.fp = cs.open('../data/record/path_extrator_'+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'.txt', 'w')
        # 加载基于Robert的微调过的文本匹配模型
        self.simmer = BertModel_Simmer().to(torch.device("cuda"))
        self.simmer.load_state_dict(torch.load("./RoBerta/models/fine_tune_best.pth.tar")["model"])
        self.simmer.eval()
        print('bert相似度匹配模型加载完成')
        print('tuples extractor loaded')

    def extract_tuples(self, candidate_entitys, question):
        ''''''
        starttime = time.time()
        candidate_tuples = {}
        entity_list = candidate_entitys.keys()  # 得到有序的实体列表
        inputs = []  # 获取所有候选路径的BERT输入
        entity_rel_dic={}
        for entity in entity_list:
            # 得到该实体的所有关系路径(1+2跳)
            if entity in self.entity2relations_dic:
                all_list = self.entity2relations_dic[entity]
            else:
                relations1hop = GetRelationPathsSingle(entity)
                relations2hop = GetRelationPaths(entity)
                relsdic_list = []
                relsdic2_list = []
                for nametype, rel in relations1hop.items():
                    relsdic_list.append(rel)
                for nametype2, rel2 in relations2hop.items():
                    relsdic2_list.append(rel2)
                if len(relsdic_list) == 1 and len(relsdic2_list) <= 1:
                    relations_list = []
                    prop_list = []
                    relation_2hop_list = []
                    rel_prop_list = []

                    for r in relsdic_list[0]['relation']:
                        relations_list.append(r)
                    for p in relsdic_list[0]['propertise']:
                        prop_list.append(p)
                    if len(relsdic2_list) == 1:
                        for i in relsdic2_list[0]['relation2hop']:
                            relation_2hop_list.append(i)
                        for j in relsdic2_list[0]['rel-prop']:
                            rel_prop_list.append(j)

                    all_list = relations_list + prop_list + relation_2hop_list + rel_prop_list

                else:
                    raise
                self.entity2relations_dic[entity] = all_list


            mention = candidate_entitys[entity][0]
            for r in all_list:
                if type(r)==str:
                    human_question = '的'.join([mention] + [r])
                else:
                    human_question='的'.join([mention]+r)
                inputs.append((question, human_question))
            entity_rel_dic[entity]=all_list
            self.entity2relations_dic[entity]=all_list
        # 将所有路径输入BERT获得分数
        print('====共有{}个候选路径===='.format(len(inputs)))
        bert_scores = []
        batch_size = 128
        Datap=DataPrecessForSentence()
        inputs_encode = Datap.get_input(data_list=inputs)
        if len(inputs) % batch_size == 0:
            num_batches = len(inputs) // batch_size
        else:
            num_batches = len(inputs) // batch_size + 1
        for i in range(num_batches):
            begin = i * batch_size
            end = min(len(inputs), (i + 1) * batch_size)
            with torch.no_grad():
                batch_seqs,batch_seq_masks,batch_seq_segments=inputs_encode[0][begin:end],inputs_encode[1][begin:end],inputs_encode[2][begin:end]
                seqs, masks, segments= batch_seqs.to(torch.device("cuda")), batch_seq_masks.to(torch.device("cuda")), batch_seq_segments.to(torch.device("cuda"))
                prediction=self.simmer(seqs,masks,segments)
            prediction=prediction.to('cpu').numpy()
            bert_scores.extend([prediction[i][1] for i in range(len(prediction))])
        print('====所有路径计算特征完毕====')

        index = 0
        for entity in entity_list:
            all_list=entity_rel_dic[entity]
            for r in all_list:
                if type(r)==str:  # 生成候选tuple
                    this_tuple = tuple([entity] + [r])
                else:
                    this_tuple = tuple([entity] +r)
                score = []  # 初始化特征
                sim2 = bert_scores[index]
                index += 1
                score.append(sim2)
                candidate_tuples[this_tuple] = score
            print('====得到实体%s的所有候选路径及其特征====' % (entity))
        print('====问题{:}的三元组特征提取共耗时{:.2}秒===='.format(question,time.time()-starttime))
        return candidate_tuples
    def GetCandidatePath(self, corpus,sim_path=1):
        '''根据mention，得到所有候选实体,进一步去知识库检索答案的候选路径
        候选路径格式为tuple(entity,relation1,relation2) 这样便于和标准答案对比
        '''
        true_num = 0
        hop2_num = 0
        hop2_true_num = 0
        all_tuples_num = 0
        l = 0
        # train_data = cs.open(sim_path,'w')
        for i in range(len(corpus)):
            dic = corpus[i]
            question = dic['question']
            if question=='男士脱发的原因和治疗':
                print(question)
            gold_tuple = dic['gold_tuple']
            if len(gold_tuple)>=4:
                print(gold_tuple)
                continue
            l += 1
            gold_entitys = dic['gold_entitys']
            candidate_entitys = dic['candidate_entity']
            if len(candidate_entitys)==0:
                print(candidate_entitys)
            print(i)
            print(question)
            candidate_tuples = self.extract_tuples(candidate_entitys, question)
            all_tuples_num += len(candidate_tuples)
            dic['candidate_tuples'] = candidate_tuples


            if_true = 0
            # train_data.write(question + '\t' + '#'.join(list(gold_tuple)) + '\t' + '1' + '\n')  # 写入句子正例
            lenth = len(candidate_tuples)

            if lenth < 11:
            # 判断gold tuple是否包含在candidate_tuples_list中
                for thistuple in candidate_tuples:
                    if len(gold_tuple) == len(set(gold_tuple) & set(thistuple)):
                        if_true = 1
                    # else:
                    #     train_data.write(question + '\t' + '#'.join(list(thistuple)) + '\t' + '0' + '\n')
            else:
                index_list = random.sample(range(0, lenth), 10)
                id = 0
                for thistuple in candidate_tuples:
                    if len(gold_tuple) == len(set(gold_tuple) & set(thistuple)):
                        if_true = 1
                        # del candidate_tuples[gold_tuple]
                        break
                for tp in candidate_tuples:
                    # if id in index_list :
                    #     train_data.write(question + '\t' + '#'.join(list(tp)) + '\t' + '0' + '\n')
                    id += 1
            if if_true == 1:
                true_num += 1
                if len(gold_tuple) <= 3 and len(gold_entitys) == 1:
                    hop2_true_num += 1
            else:
                self.fp.write(str(i) + question + '\n')
                self.fp.write(';'.join([str(item) for item in candidate_tuples.keys()]) + '\n')
                self.fp.write(str(gold_tuple) + '\n')
                self.fp.write(';'.join([str(item) for item in candidate_entitys.keys()]) + '\n')
                self.fp.write(';'.join(gold_entitys) + '\n\n')
            if len(gold_tuple) <= 3 and len(gold_entitys) == 1:
                hop2_num += 1
            corpus[i]=dic
        print('所有问题里，候选答案能覆盖标准查询路径的比例为:%.3f' % (true_num / l))
        print('单实体问题中，候选答案能覆盖标准查询路径的比例为:%.3f' % (hop2_true_num / hop2_num))
        print('平均每个问题的候选答案数量为:%.3f' % (all_tuples_num / l))
        pickle.dump(self.entity2relations_dic, open('../data/entity2relation_dic.pkl', 'wb'))
        return corpus

if __name__ == '__main__':
    inputpaths = ['../data/candidate_entitys_train.pkl','../data/candidate_entitys_valid.pkl']
    # inputpaths=['../data/candidate_entitys_valid.pkl']
    outputpaths = ['../data/candidate_tuples_train_(test).pkl','../data/candidate_tuples_valid_(test).pkl']
    # outputpaths=['../data/candidate_tuples_valid.pkl']
    PE = PathExtractor()
    for inputpath,outputpath in zip(inputpaths,outputpaths):
        #生成文本相似度训练数据
        # if inputpath=='../data/candidate_entitys_train.pkl':
        #     sim_path='./RoBerta/models/train.txt'
        # else:
        #     sim_path='./RoBerta/models/val.txt'
        corpus = pickle.load(open(inputpath,'rb'))
        # corpus = PE.GetCandidatePath(corpus,sim_path)
        corpus = PE.GetCandidatePath(corpus)
        pickle.dump(corpus,open(outputpath,'wb'))