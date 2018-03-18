import os

def merge(path):
    fs = []
    scale = []
    head = ''
    for root, paths, files in os.walk(path):
        fs = [open(os.path.join(root, file), 'r', encoding='utf-8') for file in files]
        scale = [float(file[:-4].split('_')[-1]) for file in files]

    for f in fs:
        head = f.readline()
    with open(os.path.join(path, 'merge.csv'), 'w', encoding='utf-8') as fw:
        fw.write(head)
        for line in fs[0]:
            line = line.strip().split(',')
            id = line[0]
            scores = [float(score)*scale[0] for score in line[1:]]
            for index in range(1, len(fs)):
                line = fs[index].readline().strip().split(',')
                _scores = [float(score)*scale[index] for score in line[1:]]
                scores = [scores[i]+_scores[i] for i in range(len(scores))]
            scores = list(map(str, scores))
            fw.write('{},{}\n'.format(id, ','.join(scores)))


if __name__=='__main__':
    merge('../input/merge')