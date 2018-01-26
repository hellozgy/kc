import ipdb

class ConfusionMeter(object):
    def __init__(self):
        self.matrix = [[0, 0], [0, 0]]
        self.error = [[], []]

    def add(self, predict, target, batch_size, batch_index):
        for i in range(len(predict)):
            self.matrix[predict[i]][target[i]] += 1
            if predict[i] != target[i]:
                self.error[predict[i]].append(batch_size * batch_index + i)

    def reset(self):
        self.__init__()

    def __str__(self):
        res = "{:>6}\t{:>6}\n{:>6}\t{:>6}\n".format(self.matrix[0][0], self.matrix[0][1], self.matrix[1][0], self.matrix[1][1])
        res += '<predict:0,target:1>: '
        res += ' '.join(list(map(str, self.error[0])))
        res += '\n<predict:1,target:0>: '
        res += ' '.join(list(map(str, self.error[1])))
        return res


class MulLabelConfusionMeter(object):
    def __init__(self, num_class=6):
        self.num_class = num_class
        self.matrix = [ConfusionMeter() for _ in range(self.num_class)]

    def add(self, predict, target, batch_size, batch_index):
        for i in range(self.num_class):
            self.matrix[i].add(list(map(round, predict[:, i].tolist())), list(map(int, target[:, i].tolist())), batch_size, batch_index)

    def reset(self):
        self.__init__()

    def __str__(self):
        res = '\n'.join([str(m) for m in self.matrix])
        return res



