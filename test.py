import os,sys
import pandas as pd
restore_file='checkpoint_best'
cmd_p = 'python main.py test --ngpu=0 --model=LNGRUText --id=LNGRUText_l1_c{} --res_file={} --restore_file={}'
for c in range(1, 7):
    cmd = cmd_p.format(c, 'res', restore_file)
    os.system(cmd)

res = None
cla=['toxic', 'severe_toxic','obscene','threat','insult','identity_hate']
for i in range(1, 7):
    df=pd.read_csv('./checkpoints/LNGRUText_l1_c{}/{}.csv'.format(i, 'res'),)
    df = df[['id', cla[i-1]]]
    # import ipdb;ipdb.set_trace()
    if res is None:res = df
    else:res = pd.merge(res, df, on=['id'])

res.to_csv('./checkpoints/LNGRUText_l1_c1/res_merge.csv', index=False, encoding='utf-8')


'''
python main.py test --ngpu=0 --model=LNGRUText --id=LNGRUText_l1
'''



