from neo4j import GraphDatabase

#neo4j

class KGapi:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687",auth=("neo4j","mm430422"))
        self.session = self.driver.session()
        self.nametypes=['中药名称', '中药分类名称', '其他治法名', '方剂名称', '方剂分类名称', '皮肤病名称', '证候名称', '证候分类名称', '辨证法名称']

    def GetRelationPathsSingle(self,entity):
        '''根据实体名，得到所有1跳关系路径（实体属性也视为关系）,实体名属性在各个实体中不同'''
        PathsSingle={}
        for nametype in self.nametypes:
            cql_1 = "match (a)-[r1]-() where a.{}=$name return DISTINCT type(r1)".format(nametype)
            cql_2="match (a)where a.{}=$name return properties(a)".format(nametype)
            rpaths1 = []
            props_list=[]
            res = self.session.run(cql_1, name=entity)#一个多个record组成的集合
            props=self.session.run(cql_2, name=entity)
            for record1 in res:#每个record是一个key value的有序序列
                rpaths1.append(record1['type(r1)'])
            for record2 in props:
                for prop in record2['properties(a)'].keys():
                    props_list.append(prop)
            if len(rpaths1)==0 and len(props_list)==0:
                continue
            PathsSingle[nametype]={'relation':rpaths1,'propertise':props_list}
        return PathsSingle


    def GetRelationPaths(self,entity):
        '''根据实体名，得到所有2跳关系路径，用于问题和关系路径的匹配'''
        Paths2hop={}
        for nametype in self.nametypes:
            cql_1 = "match (a)-[r1]-()-[r2]-() where a.{}=$name return DISTINCT type(r1),type(r2)".format(nametype)
            cql_2 = "match (a)-[r1]-(b)where a.{}=$name return type(r1),properties(b)".format(nametype)
            rpaths1 = []
            path2props= []
            res = self.run(cql_1,name=entity)
            resprop=self.session.run(cql_2,name=entity)
            for record1 in res:
                rpaths1.append([record1['type(r1)'],record1['type(r2)']])
            for record2 in resprop:
                for prop in record2['properties(b)'].keys():
                    path2props.append((tuple([record2['type(r1)'],prop])))
            path2props_set=[list(item) for item in set(path2props)]
            if len(rpaths1)==0 and len(path2props_set)==0:
                continue
            Paths2hop[nametype] = {'relation2hop': rpaths1, 'rel-prop': path2props_set}
        return Paths2hop


    def GetRelationNum(self,entity):
        '''根据实体名，得到与之相连的关系数量，代表实体在知识库中的流行度'''
        relcounts=0
        for nametype in self.nametypes:
            cql1= "match p=(a)-[r1]-() where a.{}=$name return count(p)".format(nametype)
            res = self.session.run(cql1,name=entity)
            ans = 0
            for record in res:
                ans = record.values()[0]
            relcounts+=ans
            return relcounts
if __name__ == '__main__':
    kg=KGapi()
    a=kg.GetRelationNum('天疤疮')
    print(a)