res = []
def f(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            t = len(line.strip())
            res.append(t)
            if t==1:print(line)

f('../input/train_data_bpe_train.csv')
f('../input/train_data_bpe_val.csv')
f('../input/train_data_bpe_test.csv')
f('../input/test_data_bpe.csv')

print(len(res))
print(min(res))
