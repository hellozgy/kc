# coding: utf-8

import pandas as pd
import re, string, os
import unicodedata
import ipdb

# 1.将数据标签分开，并将内容转化为一行
outdata_oneline = '../input/outdata_oneline.csv'

#2.去除不可见字符，将中文标点转化为英文标点
# moses_dir = 'd:/git/mosesdecoder-master/scripts/tokenizer'
moses_dir = '/users2/hpzhao/gyzhu/git/mosesdecoder-master/scripts/tokenizer'
cmd = 'cat ../input/outdata_oneline.csv | perl {}/remove-non-printing-char.perl | perl {}/replace-unicode-punctuation.perl > ' \
      '../input/outdata_oneline_moses.csv'.format(moses_dir, moses_dir)
os.system(cmd)

# 3. 删除首尾标点和重复标点
outdata_data_src = '../input/outdata_oneline_moses.csv'
outdata_data_dst = '../input/outdata_oneline_moses_punc.csv'

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
            tag = 0 in set([0 if t in badword else 1 for t in line[index: index+n]])
            if index+n <= len(line) and (m_word in badword or m_word2 in badword) and abs(len(line[index])-len(line[index+n-1]))<=1 and not tag:
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
            # for _ in range(5):
            #     tt = re.sub(r'(\w+)\.(\w+)', r'\1\2', tt)
            if len(tt)>=6:
                tt = re.sub(r'([!,\.?:;"&\-_=[\]|()/])\1*', r' \1 ', tt).split()
                res.extend(tt)
            else:
                res.append(tt)
    line = ' '.join(res)
    return line

special_tag = set(['IP', 'EM', 'URL'])

def filter(src, dst):
    f = open(src, 'r', encoding='utf-8')
    fw = open(dst, 'w', encoding='utf-8')
    for line in f.readlines():
        line = re.sub('\\s+', ' ', line)
        line = line.strip().lower()
        line = unicodedata.normalize('NFKC', line)
        line = line.strip(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’ ])')
        line = ' '+line +' '
        line = re.sub(r'\d+:\d+', ' ', line)  # 删除时间
        line = re.sub(r'(\d+\.\d+\.\d+\.\d+)\1*', r' IP \1 ', line)  # 处理IP
        line = re.sub(r' \W*\d+\.?\d*\W* ', ' ', ' '+line+' ')  # 删除纯数字
        line = re.sub(r' \W*\d+/?\d*\W* ', ' ',  line)  # 删除纯数字

        line = line.replace('"', ' ').replace('\' ', ' ').replace('~',' ').replace('^',' ').replace(' ur ', ' your ').replace('. ', ' . ').replace(' .',' . ')
        line = re.sub(r'([!"#$%&\'()*+,\-\./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’ \n]+)\1{2,}', r'\1', line)
        line = re.sub(r'\s([!"#$%&\'()*+,\-\./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’])\1+\s', r' \1 ', line)
        # detect email adresses
        line = re.sub(r'([\w\-][\w\-\.]+@[\w\-][\w\-\.]+\.[a-zA-Z]{1,4})\1*', r' EM \1 ', line)
        # detect urls, retain host
        line = re.sub(r'http(s)*://([\w\d\.\-]+)(/\S+)*', r' URL \2 ', line)
        line = re.sub(r'(www\.[\w\d\.\-]+)(/\S+)*', r' URL \1 ', line)

        line = re.sub(r'(.)\1{3,}', r'\1', line)
        line = re.sub(r'(..)\1{2,}', r'\1', line)
        line = re.sub(r'(...)\1{2,}', r'\1', line)
        line = re.sub(r'(....)\1{1,}', r'\1', line)
        line = re.sub(r'(.....)\1{1,}', r'\1', line)
        line = re.sub(r'(......)\1{1,}', r'\1', line)
        line = re.sub(r'(.......)\1{1,}', r'\1', line)
        line = re.sub(r'(........)\1{1,}', r'\1', line)
        line = re.sub(r'(.........)\1{1,}', r'\1', line)

        line = re.sub('\\s+', ' ', line)
        line = re.sub(r'((\S+\s+){1})\1*', r'\1', line + ' ')
        line = re.sub(r'((\S+\s+){2})\1*', r'\1', line + ' ')
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
        line = merge(line)
        line = re.sub('\\s+', ' ', line)
        line = re.sub(r'((\S+\s+){1})\1*', r'\1', line + ' ')
        line = re.sub('\\s+', ' ', line)
        line = line.strip()
        fw.write(line+ '\n')
        fw.flush()
    f.close()
    fw.close()

filter(outdata_data_src, outdata_data_dst)
print(3)

cmd = "export CLASSPATH='/users2/hpzhao/gyzhu/git/stanford-postagger-2017-06-09/stanford-postagger.jar:$CLASSPATH'"
os.system(cmd)
#4.分词
cmd = "java edu.stanford.nlp.process.PTBTokenizer -preserveLines " \
      "-options 'ptb3Escaping=false," \
      "tokenizePerLine=true," \
      "normalizeFractions=false," \
      "normalizeParentheses=false," \
      "normalizeOtherBrackets=false," \
      "latexQuotes=false," \
      "ptb3Ellipsis=false," \
      "ptb3Dashes=false," \
      "escapeForwardSlashAsterisk=false," \
      "strictTreebank3=true' " \
      "< ../input/{}_oneline_moses_punc.csv > ../input/{}_data.csv"
os.system(cmd.format('outdata', 'outdata'))

#5. BPE子字
subword_path = '/users2/hpzhao/gyzhu/git/subword-nmt'
cmd = 'python {}/learn_joint_bpe_and_vocab.py --input ../input/train_data.csv ../input/test_data.csv ' \
      '-s 50000 -o ../input/train_test.codes --write-vocabulary ../input/train.vocab ../input/test.vocab'.format(subword_path)
os.system(cmd)
cmd = 'python {}/apply_bpe.py -c ../input/train_test.codes --vocabulary ../input/train.vocab --vocabulary-threshold 5 -s "" ' \
      ' < ../input/outdata_data.csv > ../input/outdata_data_bpe.csv'.format(subword_path)
os.system(cmd)

# 5.2数据去重
train_set = set()
with open('../input/train_data_bpe.csv', 'r', encoding='utf-8') as f1:
    for line in f1:
        train_set.add(line.strip())

with open('../input/outdata_data_bpe.csv', 'r', encoding='utf-8') as f1,\
    open('../input/outdata_label.csv', 'r', encoding='utf-8') as f2,\
    open('../input/outdata_data_bpe_final.csv', 'w', encoding='utf-8') as f3,\
    open('../input/outdata_label_final.csv', 'w', encoding='utf-8') as f4:
    for data, label in zip(f1, f2):
        data = data.strip()
        label = label.strip()
        if data=='' or data in train_set:continue
        f3.write(data+'\n')
        f4.write(label+'\n')
print(5.2)
