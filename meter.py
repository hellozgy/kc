import ipdb

class ConfusionMeter(object):
    def __init__(self, label, simple=False):
        self.label = label
        self.simple = simple
        self.target = []
        self.predict = []

    def add(self, predict, target):
        self.target.extend(target)
        self.predict.extend(predict)

    def reset(self):
        self.__init__(self.label)

    def get(self):
        n = sum(self.target)
        d = [(p, t, num + 1) for num, (p, t) in enumerate(zip(self.predict, self.target))]
        d = sorted(d, key=lambda x: x[0], reverse=True)
        p1t0 = []
        recall = 0
        for num, (p, t, index) in enumerate(d):  # 召回率
            if num == n: break
            if t == 0:
                p1t0.append(index)
            else:
                recall += 1
        recall = recall / n
        return recall

    def __str__(self):
        n = sum(self.target)
        d = [(p, t, num+1) for num, (p, t) in enumerate(zip(self.predict, self.target))]
        d = sorted(d, key=lambda x:x[0], reverse=True)
        p1t0 = []
        recall = 0
        for num, (p, t, index) in enumerate(d): #召回率
            if num==n:break
            if t==0:
                p1t0.append(index)
            else:
                recall+=1
        recall = recall/n
        p1t0 = ','.join([str(t) for t in sorted(p1t0)])
        p0t1 = []
        for num, (p, t, index) in enumerate(d[n:]):
            if t==1:
                p0t1.append(index)
        p0t1 = ','.join([str(t) for t in sorted(p0t1)])
        res = "类别:{}\t精度:{}".format(self.label, recall)
        if not self.simple:
            res += '\n<predict:0,target:1>: {}'.format(p0t1)
            res += '\n<predict:1,target:0>:{} '.format(p1t0)
        return res


class MulLabelConfusionMeter(object):
    def __init__(self, simple=False, num_class=6):
        self.labels = ['toxic','severe','obscene','threat','insult','identity']
        self.num_class = num_class
        self.matrix = [ConfusionMeter(self.labels[i], simple) for i in range(self.num_class)]

    def add(self, predict, target, batch_size, batch_index):
        '''
        :param predict: batch_size * 6（numpy）
        :param target: :batch_size * 6(numpy)
        :param batch_size:
        :param batch_index:第几个batch，从0开始
        :return:
        '''
        for i in range(self.num_class):
            self.matrix[i].add(predict[:, i].tolist(), list(map(int, target[:, i].tolist())))

    def reset(self):
        self.__init__()

    def get(self):
        return [m.get() for m in self.matrix]

    def __str__(self):
        res = '\n'.join([str(m) for m in self.matrix])
        return res



