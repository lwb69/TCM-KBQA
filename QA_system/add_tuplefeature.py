import pickle


def add(corpus):
    for i in range(len(corpus)):
        dic = corpus[i]
        question = dic['question']
        gold_tuple = dic['gold_tuple']
        gold_entitys = dic['gold_entitys']
        candidate_entitys = dic['candidate_entity']
        if len(candidate_entitys) == 0:
            print(candidate_entitys)
        candidate_tuples=dic['candidate_tuples']
        entity_list = candidate_entitys.keys()
        for entity in entity_list:
            print(entity)
            for tuple in candidate_tuples:
                candidate_tuples[tuple]=candidate_tuples[tuple]
        score = [entity] + candidate_entitys[entity]  # 初始化特征
        sim2 = bert_scores[index]
        index += 1
        score.append(sim2)
        candidate_tuples[this_tuple] = score
inputpaths = ['../data/candidate_tuples_train.pkl','../data/candidate_tuples_valid.pkl']
outputpaths=['../data/candidate_tuples_train.pkl','../data/candidate_tuples_valid.pkl']
for inputpath, outputpath in zip(inputpaths, outputpaths):
    corpus = pickle.load(open(inputpath, 'rb'))
    corpus = add(corpus)
    # pickle.dump(corpus, open(outputpath, 'wb'))