from qa.system.QA_system.mention_extrator import MentionExtractor
from qa.system.QA_system.entitylink import EntityExtractor
from qa.system.QA_system.path_extrator import PathExtractor
from qa.system.QA_system.KG import KGapi
from neo4j import GraphDatabase
import pickle

class AnswerBot():
    def __init__(self):
        self.me = MentionExtractor()
        self.el = EntityExtractor()
        self.pt = PathExtractor()
        self.driver = GraphDatabase.driver("bolt://localhost:7687",auth=("neo4j","mm430422"))
        self.session = self.driver.session()
        self.KGapi=KGapi()
        self.nametypes=['中药名称', '中药分类名称', '其他治法名', '方剂名','方剂名称', '方剂分类名称', '皮肤病名称', '证候名称', '证候分类名称', '辨证法名称']
    def answer_main(self, question):
        '''
               输入问题，依次执行：
               抽取实体mention、生成候选实体、生成候选查询路径（单实体单跳）、生成候选查询路径与问题间的语义相似度，候选查询路径过滤
               使用top1的候选查询路径检索答案并返回
               input:
                   question : python-str
               output:
                   answer : python-list, [str]
               '''
        dic = {}
        dic['question'] = question
        # print(question)

        mentions = self.me.extract_mentions(question)
        dic['mentions'] = mentions
        print('====实体mention为====')
        print(mentions.keys())

        entity = self.el.extract_subject(entity_mentions=mentions, question=question)
        dic['link_entity'] = entity
        print('====链接到的实体为====')
        print(entity.keys())
        if len(entity) == 0:
            return ''
        for enti in entity.keys():
            type=self.KGapi.Gettypeofentity(enti)
            entitytuple=(enti,type)
        #单纯搜索美容方名字，返回美容方功效和治法属性

        if len(entity.keys())==1:
            for enti in entity.keys():
                if enti==question:
                    answer=self.anserforone(enti)
                    return answer


        #路径抽取
        tuples = self.pt.extract_tuples(candidate_entitys=entity, question=question)
        dic['tuples'] = tuples
        if len(tuples) == 0:
            return ''
        score_max=0
        for tuple in tuples:
            print("召回的路径有",tuple,tuples[tuple])
            current_score=tuples[tuple][0]
            if current_score>score_max:
                score_max=current_score
                best_tuple=tuple
        print ('====最终候选查询路径为====')
        print(best_tuple)

        # 生成cypher语句并查询
        #单挑路径
        if len(best_tuple) == 2:
            PathsSingle = {}
            for nametype in self.nametypes:
                sql1 = "match (a)-[r1]-(b) where a.{}=$ename and type(r1)=$rname return b,labels(b)".format(nametype)
                sql2 = "match (a)where a.{}=$name return a.{}".format(nametype, best_tuple[1])
                # if nametype=='方剂名称':
                #     sql1 = "match (a:`美容方`)-[r1]-(b) where a.{}=$ename and type(r1)=$rname return b".format(nametype)
                #     sql2 = "match (a:`美容方`)where a.{}=$name return a.{}".format(nametype, best_tuple[1])

                res1 = self.session.run(sql1, ename=best_tuple[0], rname=best_tuple[1])
                res2= self.session.run(sql2, name=best_tuple[0])
                entity_list = []
                props_list = []
                for record1 in res1:  # 每个record是一个key value的有序序列
                    for name in self.nametypes:
                        ans = []
                        ans.append(record1['b'][name])
                        ans.append(record1['labels(b)'][0])
                        if ans[0]:
                            entity_list.append(ans)
                            break
                for record2 in res2:
                    prop=record2['a.{}'.format(best_tuple[1])]
                    props_list.append(prop)
                if len(entity_list) == 0 and len(props_list) == 0:
                    continue
                else:
                    PathsSingle[nametype]={'entity':entity_list,'propertise':props_list}
                    break
            print(PathsSingle)
        #两跳路径
        if len(best_tuple) == 3:
            PathsSingle = {}
            for nametype in self.nametypes:
                cql_1 = "match (a)-[r1]-()-[r2]-(b) where a.{}=$name and type(r1)=$r1name and type(r2)=$r2name return b,labels(b)".format(
                    nametype)
                cql_2 = "match (a)-[r1]-(b)where a.{}=$name and type(r1)=$r1name return b.{}".format(nametype,
                                                                                                     best_tuple[2])
                # if nametype=='方剂名称':
                #     cql_1 = "match (a:`美容方`)-[r1]-()-[r2]-(b) where a.{}=$name and type(r1)=$r1name and type(r2)=$r2name return b".format(
                #         nametype)
                #     cql_2 = "match (a:`美容方`)-[r1]-(b)where a.{}=$name and type(r1)=$r1name return b.{}".format(nametype,
                #                                                                                          best_tuple[2])

                res1 = self.session.run(cql_1, name=best_tuple[0],r1name=best_tuple[1],r2name=best_tuple[2])
                res2 = self.session.run(cql_2, name=best_tuple[0],r1name=best_tuple[1])
                entity_list = []
                props_list = []
                for record1 in res1:
                    for name in self.nametypes:
                        ans = []
                        ans.append(record1['b'][name])
                        ans.append(record1['labels(b)'][0])
                        if ans[0]:
                            entity_list.append(ans)
                            break
                for record2 in res2:
                    prop = record2['b.{}'.format(best_tuple[2])]
                    props_list.append(prop)
                if len(entity_list) == 0 and len(props_list) == 0:
                    continue
                else:
                    PathsSingle[nametype]={'entity':entity_list,'propertise':props_list}
                    break
            print(PathsSingle)
        #生成答案
        for nt in PathsSingle:
            entity=PathsSingle[nt]['entity']
            proper=PathsSingle[nt]['propertise']
            if len(entity)==0:
                if len(proper)==0:
                    answer=[]
                else:
                    answer=proper
            else:
                answer=entity
        print('answer值为：',answer)

        # if len(answer)==0:
        #     answer=''
        # else:
        #     answer=','.join(answer)
        # if type(answer)!=str:
        #     raise
        return (entitytuple,answer)

    def anserforone(self,enty):
        for nametype in self.nametypes:
            if nametype == '中药名称':
                sql1 = "match (a)where a.{}=$name return a.{}".format('中药名称', '中药简介')
                res1 = self.session.run(sql1, name=enty)
                for record1 in res1:
                    prop1 = record1['a.{}'.format('中药简介')]
                    if prop1 !=None and prop1 !='':
                        answer = enty + ' : ' + prop1
                        return answer
            if nametype=='皮肤病名称':
                sql1 = "match (a)where a.{}=$name return a.{}".format('皮肤病名称', '皮肤病定义')
                res1 = self.session.run(sql1, name=enty)
                for record1 in res1:
                    prop1 = record1['a.{}'.format('皮肤病定义')]
                    if prop1 != None and prop1 != '':
                        answer = enty + '的定义:  ' + prop1
                        return answer
            if nametype=='方剂名称':
                sql1 = "match (a)where a.{}=$name return a.{}".format('方剂名称', '功效')
                res1 = self.session.run(sql1, name=enty)
                for record1 in res1:
                    prop1 = record1['a.{}'.format('功效')]
                    if prop1 != None and prop1 != '':
                        answer = enty + '的功效:  ' + prop1
                        return answer

    def add_answers_to_corpus(self, corpus):
        for i in range(len(corpus)):
            sample = corpus[i]
            question = sample['question']
            ans = self.answer_main(question)
            sample['predict_ans'] = ans
        return corpus



if __name__ == '__main__':
    ansbot = AnswerBot()
    while True:
        question=input()
        answer_list = ansbot.answer_main(question)
        if len(answer_list)==0:
            answer=''
        else:
            answer=','.join(answer_list)
        print("答案为:",answer)
    # corpus = pickle.load(open('../data/candidate_tuples_valid.pkl', 'rb'))
    # corpus = ansbot.add_answers_to_corpus(corpus)
    # outputpath='../data/valid_answers.pkl'
    # pickle.dump(corpus, open(outputpath, 'wb'))
    # ave_f = 0.0
    # for i in range(len(corpus)):
    #     sample = corpus[i]
    #     gold_ans = sample['answer']
    #     gold_ans=list(map(eval,gold_ans))
    #     pre_ans = sample['predict_ans']
    #     true = len(set(gold_ans).intersection(set(pre_ans)))
    #     p = true / len(set(pre_ans))
    #     r = true / len(set(gold_ans))
    #     try:
    #         f = 2 * p * r / (p + r)
    #     except:
    #         f = 0.0
    #     ave_f += f
    # ave_f /= len(corpus)
    #
    # print(ave_f)
