import thulac
segger = thulac.thulac(seg_only=True)
class RunningAverage:
    """A simple class that maintains the running average of a quantity
    记录平均损失

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def ComputeEntityFeatures(question,entity,relations):
    '''
        抽取每个实体或属性值2hop内的所有关系，来跟问题计算各种相似度特征
        input:
            question: python-str
            entity: python-str <entityname>
            relations: python-list rname and property
        output：
            [word_overlap,char_overlap]
        '''
    # 得到主语-谓词的tokens及chars
    p_tokens = []
    for p in relations:
        p_tokens.extend(segger.cut(p))
    p_tokens = [token[0] for token in p_tokens]
    p_chars = [char for char in ''.join(p_tokens)]

    q_tokens = segger.cut(question)
    q_tokens = [token[0] for token in q_tokens]
    q_chars = [char for char in question]

    e_tokens = segger.cut(entity)
    e_tokens = [token[0] for token in e_tokens]
    e_chars = [char for char in entity]

    qe_feature = features_from_two_sequences(q_tokens, e_tokens) + features_from_two_sequences(q_chars, e_chars)
    qr_feature = features_from_two_sequences(q_tokens, p_tokens) + features_from_two_sequences(q_chars, p_chars)
    return qe_feature + qr_feature

def features_from_two_sequences(s1,s2):
    #overlap
    overlap = len(set(s1)&(set(s2)))
    #集合距离
    jaccard = len(set(s1)&(set(s2))) / len(set(s1)|(set(s2)))
    return [overlap,jaccard]