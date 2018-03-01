import ipdb

class ConfusionMeter(object):
    def __init__(self, label):
        self.label = label
        self.matrix = [[0, 0], [0, 0]]
        self.error = [[], []]

    def add(self, predict, target, batch_size, batch_index):
        for i in range(len(predict)):
            self.matrix[predict[i]][target[i]] += 1
            if predict[i] != target[i]:
                self.error[predict[i]].append(batch_size * batch_index + i + 1)

    def reset(self):
        self.__init__(self.label)

    def __str__(self):
        res = "{}:{:>6} {:>6} {:>6}  {:>6}  precision:{:>2}%\trecall:{:>2}%\n".format(
            self.label, self.matrix[0][0], self.matrix[0][1], self.matrix[1][0], self.matrix[1][1],
            int(self.matrix[1][1]/sum(self.matrix[1])*100), int(self.matrix[1][1]/(self.matrix[0][1]+self.matrix[1][1])*100))
        res += '<predict:0,target:1>: '
        res += ' '.join(list(map(str, self.error[0])))
        res += '\n<predict:1,target:0>: '
        res += ' '.join(list(map(str, self.error[1])))
        return res


class MulLabelConfusionMeter(object):
    def __init__(self, num_class=6):
        self.labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        self.num_class = num_class
        self.matrix = [ConfusionMeter(self.labels[i]) for i in range(self.num_class)]

    def add(self, predict, target, batch_size, batch_index):
        '''
        :param predict: batch_size * 6（numpy）
        :param target: :batch_size * 6(numpy)
        :param batch_size:
        :param batch_index:第几个batch，从0开始
        :return:
        '''
        for i in range(self.num_class):
            self.matrix[i].add(list(map(round, predict[:, i].tolist())), list(map(int, target[:, i].tolist())), batch_size, batch_index)

    def reset(self):
        self.__init__()

    def __str__(self):
        res = '\n'.join([str(m) for m in self.matrix])
        return res



