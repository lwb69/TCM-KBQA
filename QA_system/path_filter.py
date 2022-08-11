import pickle
from sklearn import linear_model
import numpy as np
from utils import *

def GetData(corpus):
    '''为验证集验证模型使用的数据
    X : numpy.array, (num_sample,num_feature)
    Y : numpy.array, (num_sample,1)
    samples : python-list,(num_sample,)
    ans : python-list, (num_question,num_answer)
    question2sample : python-dict, key:questionindex , value:sampleindexs
    '''
    X = []
    Y = []
    samples = []
    ans = []
    gold_tuples = []
    question2sample = {}

    sample_index = 0
    true_num = 0
    hop2_num = 0
    hop2_true_num = 0
    for i in range(len(corpus)):
        candidate_tuples = corpus[i]['candidate_tuples']
        gold_tuple = corpus[i]['gold_tuple']
        gold_entitys = corpus[i]['gold_entitys']
        answer = corpus[i]['answer']
        q_sample_indexs = []

        for t in candidate_tuples:
            features = candidate_tuples[t]
            if len(gold_tuple) == len(set(gold_tuple).intersection(set(t))):
                X.append([features[2]])
                Y.append([1])
            else:
                X.append([features[2]])
                Y.append([0])

            samples.append(t)
            q_sample_indexs.append(sample_index)
            sample_index += 1
        ans.append(answer)
        gold_tuples.append(gold_tuple)
        question2sample[i] = q_sample_indexs

        if_true = 0
        # 判断gold tuple是否包含在候选tuples中
        for thistuple in candidate_tuples:
            if cmp(thistuple, gold_tuple) == 0:
                if_true = 1
                break
        # 判断单实体问题中，可召回的比例
        if if_true == 1:
            true_num += 1
            if len(gold_tuple) <= 3 and len(gold_entitys) == 1:
                hop2_true_num += 1
        if len(gold_tuple) <= 3 and len(gold_entitys) == 1:
            hop2_num += 1

    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    print('单实体问题中，候选答案可召回的的比例为:%.3f' % (hop2_true_num / hop2_num))
    print('候选答案能覆盖标准查询路径的比例为:%.3f' % (true_num / len(corpus)))
    return X, Y, samples, ans, gold_tuples, question2sample


def GetData_train(corpus):
    '''
        为训练集的候选答案生成逻辑回归训练数据，由于正负例非常不均衡，对于负例进行0.05的采样
        '''
    X = []
    Y = []
    true_num = 0
    hop2_num = 0
    hop2_true_num = 0
    for i in range(len(corpus)):
        candidate_tuples = corpus[i]['candidate_tuples']  # 字典
        gold_tuple = corpus[i]['gold_tuple']
        gold_entitys = corpus[i]['gold_entitys']
        neg_n = 0
        for t in candidate_tuples:
            features = candidate_tuples[t]
            if len(gold_tuple) == len(set(gold_tuple).intersection(set(t))):
                X.append([features[2]])
                Y.append([1])
            else:
                if neg_n < 5:  # 得是0.05吧
                    X.append([features[0]])
                    Y.append([0])
                    neg_n += 1

        if_true = 0  # 判断答案是否召回
        for thistuple in candidate_tuples:
            if cmp(thistuple, gold_tuple) == 0:
                if_true = 1
                break
        if if_true == 1:
            true_num += 1
            if len(gold_tuple) <= 3 and len(gold_entitys) == 1:
                hop2_true_num += 1
        if len(gold_tuple) <= 3 and len(gold_entitys) == 1:
            hop2_num += 1

    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    print('单实体问题中，候选答案可召回的的比例为:%.3f' % (hop2_true_num / hop2_num))
    print('候选答案能覆盖标准查询路径的比例为:%.3f' % (true_num / len(corpus)))
    return X, Y

if __name__ == '__main__':
    inputpaths=['../data/candidate_tuples_feature_train.pkl','../data/candidate_tuples_feature_valid.pkl']
    outputpaths=['../data/candidate_tuples_feature_train.pkl','../data/candidate_tuples_feature_valid.pkl']

    for inputpath,outputpath in zip(inputpaths,outputpaths):
        corpus= pickle.load(open(inputpath, 'rb'))
        corpus=GetTuplefeature(corpus)
        pickle.dump(corpus, open(outputpath, 'wb'))

    # x_train, y_train = GetData_train(train_corpus)
    # x_valid, y_valid, samples_valid, ans_valid, gold_tuples_valid, question2sample_valid = GetData(valid_corpus)
