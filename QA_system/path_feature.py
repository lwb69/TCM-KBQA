import pickle

from utils import *

def GetTuplefeature(corpus):
    '''
    为path生成相关特征
    :param corpus:
    :return: corpus
    '''
    num=0
    for i in range(len(corpus)):
        dic = corpus[i]
        question = dic['question']
        gold_tuple = dic['gold_tuple']
        if len(gold_tuple) >= 4:
            # print(gold_tuple)
            continue
        candidate_tuples = dic['candidate_tuples']
        for candidata_tuple in candidate_tuples:
            candidate_path=''.join(candidata_tuple)
            features=features_from_two_sequences(candidate_path,question)
            candidate_tuples[candidata_tuple]=candidate_tuples[candidata_tuple]+features
            if gold_tuple ==candidata_tuple:
                print(candidate_tuples[candidata_tuple])

        dic['candidate_tuples']=candidate_tuples
        num+=1
        print(gold_tuple)
    print(num)
if __name__ == '__main__':
    inputpaths=['../data/candidate_tuples_train.pkl','../data/candidate_tuples_valid.pkl']
    outputpaths=['../data/candidate_tuples_feature_train.pkl','../data/candidate_tuples_feature_valid.pkl']

    for inputpath,outputpath in zip(inputpaths,outputpaths):
        corpus= pickle.load(open(inputpath, 'rb'))
        corpus=GetTuplefeature(corpus)
        pickle.dump(corpus, open(outputpath, 'wb'))