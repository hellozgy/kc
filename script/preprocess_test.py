# coding: utf-8

import pandas as pd
import re, string, os
import unicodedata, nltk
import ipdb
# stemmer = nltk.stem.PorterStemmer()
'''
# 1.将数据标签分开，并将内容转化为一行
test = '../test_all_english.csv'
test_oneline = '../input/test_oneline2.csv'

with open(test, 'r', encoding='utf-8') as f, open(test_oneline, 'w', encoding='utf-8') as fw:
    for line in f:
        line = line.strip()
        line = ','.join(line.split(',')[1:])
        fw.write(line+'\n')
print('finish 1')
#2.去除不可见字符，将中文标点转化为英文标点
# moses_dir = 'd:/git/mosesdecoder-master/scripts/tokenizer'
moses_dir = 'e:/git/mosesdecoder-master/scripts/tokenizer'
cmd = 'cat ../input/test_oneline2.csv | perl {}/remove-non-printing-char.perl | perl {}/replace-unicode-punctuation.perl > ' \
      '../input/test_oneline_moses2.csv'.format(moses_dir, moses_dir)
os.system(cmd)
print('finish 2')
#3. 删除首尾标点和重复标点
test_data_src = '../input/test_oneline_moses2.csv'
test_data_dst = '../input/test_oneline_moses_punc2.csv'
badword = set()
with open('../input/badword.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if len(line)>=3:
            badword.add(line)
        if len(line.replace(' ',''))>=3:
            badword.add(line.replace(' ',''))
        if len(re.sub(r'([^a-zA-Z])\1*', r'', line))>=3:
            badword.add(re.sub(r'([^a-zA-Z])\1*', r'', line))

def merge(line):
    line = line.split()
    for n in range(10, 0, -1):
        index = 0
        res = []
        while index < len(line):
            m_word = ''.join(line[index:index + n])
            m_word2 = re.sub(r'([^a-zA-Z])\1*', r'', m_word)
            if index+n <= len(line) and (m_word in badword or m_word2 in badword):
                res.append(m_word if m_word2 not in badword else m_word2)
                index += n
            else:
                res.append(line[index])
                index += 1
        line = res
    res = []

    for index,tt in enumerate(line):
        if (index>0 and line[index-1] in special_tag) or tt in badword:
            res.append(tt)
        else:
            for _ in range(5):
                tt = re.sub(r'(\w+)\.(\w+)', r'\1\2', tt)
            if len(tt)>=6:
                tt = re.sub(r'([!,\.?:;"&\-_=[\]|()/])\1*', r' \1 ', tt).split()
                res.extend(tt)
            else:
                res.append(tt)
    line = ' '.join(res)
    return line

special_tag = set(['TIME','IP', 'EM', 'URL'])
def filter(src, dst):
    f = open(src, 'r', encoding='utf-8')
    fw = open(dst, 'w', encoding='utf-8')
    for line in f.readlines():
        line = line.strip().lower()
        line = re.sub('\\s+', ' ', line)
        line = unicodedata.normalize('NFKC', line)
        line = re.sub(r'\d+:\d+', ' TIME ', line)  # 删除时间
        line = re.sub(r'(\d+\.\d+\.\d+\.\d+)\1*', r' IP \1 ', line)  # 处理IP
        line = re.sub(r' \d+\.?\d* ', ' DD ', line) # 删除纯数字

        line = line.replace('"',' ').replace(' \'','').replace('\' ','').replace('\'',' \'')

        # detect email adresses
        line = re.sub(r'([\w\-][\w\-\.]+@[\w\-][\w\-\.]+\.[a-zA-Z]{1,4})\1*', r' EM \1 ', line)
        # detect urls, retain host
        line = re.sub(r'http(s)*://([\w\d\.\-]+)(/\S+)*', r' URL \2 ', line)
        line = re.sub(r'(www\.[\w\d\.\-]+)(/\S+)*', r' URL \1 ', line)
        line = re.sub(r'[~^]', ' ', line)
        line = re.sub(r'(.)\1{3,}', r'\1', line)
        line = re.sub(r'(..)\1{2,}', r'\1', line)
        line = re.sub(r'(...)\1{2,}', r'\1', line)
        line = re.sub(r'(....)\1{1,}', r'\1', line)
        line = re.sub(r'(.....)\1{1,}', r'\1', line)
        line = re.sub(r'(......)\1{1,}', r'\1', line)
        line = re.sub(r'(.......)\1{1,}', r'\1', line)
        line = re.sub(r'(........)\1{1,}', r'\1', line)
        line = re.sub(r'(.........)\1{1,}', r'\1', line)
        line = line.strip(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’ \n])')
        # line = re.sub(fr'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’ \n])\1*', r'\1', line)
        line = re.sub(r'([!"#$%&\'()*+,\-\./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’ \n])\1{2,}', r'\1', line)
        line = line.replace(' ur ', ' your ')
        # line = re.sub(r'([^*`\'a-zA-Z])\1*', r' \1 ', line)
        line = re.sub('\\s+', ' ', line)
        line = re.sub(r'((\S+\s+){1})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){2})\1*', r'\1', line+' ')
        line = re.sub(r'((\S+\s+){3})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){4})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){5})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){6})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){7})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){8})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){9})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){10})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){11})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){12})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){13})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){14})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){15})\1*', r'\1', line + ' ')
        # for _ in range(5):
        #     line = re.sub(r'(\w+)\.(\w+)', r'\1\2', line)
        line = merge(line)
        line = re.sub('\\s+', ' ', line)
        line = line.strip()
        fw.write(line+'\n')
        fw.flush()
    f.close()
    fw.close()
            
filter(test_data_src, test_data_dst)
print('finish 3')
# 3.2数据去重
import shutil
shutil.copy(test_data_dst, '../input/test_data2.csv')
print('finish 3.2')
'''
#5. BPE子字
subword_path = '/home/zgy/git/subword-nmt'
cmd = 'python {}/learn_joint_bpe_and_vocab.py --input ../input/train_data.csv ../input/test_data.csv ' \
      '-s 50000 -o ../input/train_test.codes --write-vocabulary ../input/train.vocab ../input/test.vocab'.format(subword_path)
os.system(cmd)
cmd = 'python {}/apply_bpe.py -c ../input/train_test.codes --vocabulary ../input/train.vocab --vocabulary-threshold 5 -s "" ' \
      ' < ../input/test_data2.csv > ../input/test_data2_bpe.csv'.format(subword_path)
os.system(cmd)
print('finish 5')