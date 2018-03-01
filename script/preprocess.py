# coding: utf-8

import pandas as pd
import re, string, os
import unicodedata, nltk
import ipdb
# stemmer = nltk.stem.PorterStemmer()

# 1.将数据标签分开，并将内容转化为一行
train = '../input/train.csv'
test = '../input/test.csv'
train_oneline = '../input/train_oneline.csv'
train_label = '../input/train_label_tmp.csv'
train_label_final = '../input/train_label.csv'
test_oneline = '../input/test_oneline.csv'
test_label = '../input/test_label.csv'

train = pd.read_csv(train)
test = pd.read_csv(test)
COMMENT = 'comment_text'
train[COMMENT].dropna()

train[COMMENT].str.replace('\n',' ').to_csv(train_oneline, header=False, index=False, encoding='utf-8')
del train[COMMENT]
train.to_csv(train_label, header=False, index=False, encoding='utf-8')
test[COMMENT].str.replace('\n', ' ').to_csv(test_oneline, header=False, index=False, encoding='utf-8')
del test[COMMENT]
test.to_csv(test_label, header=False, index=False, encoding='utf-8')
print('finish 1')
#2.去除不可见字符，将中文标点转化为英文标点
# moses_dir = 'd:/git/mosesdecoder-master/scripts/tokenizer'
moses_dir = '/home/zgy/git/mosesdecoder-master/scripts/tokenizer'
cmd = 'cat ../input/train_oneline.csv | perl {}/remove-non-printing-char.perl | perl {}/replace-unicode-punctuation.perl > ' \
      '../input/train_oneline_moses.csv'.format(moses_dir, moses_dir)
os.system(cmd)
cmd = 'cat ../input/test_oneline.csv | perl {}/remove-non-printing-char.perl | perl {}/replace-unicode-punctuation.perl > ' \
      '../input/test_oneline_moses.csv'.format(moses_dir, moses_dir)
os.system(cmd)
print('finish 2')
#3. 删除首尾标点和重复标点
train_data_src = '../input/train_oneline_moses.csv'
train_data_dst = '../input/train_oneline_moses_punc.csv'
test_data_src = '../input/test_oneline_moses.csv'
test_data_dst = '../input/test_oneline_moses_punc.csv'
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
            
filter(train_data_src, train_data_dst)
filter(test_data_src, test_data_dst)
print('finish 3')
# 3.2数据去重
with open(train_data_dst, 'r', encoding='utf-8') as f1, open(train_label, 'r', encoding='utf-8') as f2,\
        open('../input/train_data.csv', 'w', encoding='utf-8') as fw1, open(train_label_final, 'w', encoding='utf-8') as fw2:
    res = set()
    for data, label in zip(f1, f2):
        data = data.strip()
        if data=='':continue
        label = label.strip()
        key = ','.join(label.split(',')[1:])+data
        if key not in res:
            res.add(key)
            fw1.write(data+'\n')
            fw2.write(label+'\n')

import shutil
shutil.copy(test_data_dst, '../input/test_data.csv')
print('finish 3.2')
#4.分词
# cmd = "java edu.stanford.nlp.process.PTBTokenizer -preserveLines " \
#       "-options 'ptb3Escaping=false," \
#       "tokenizePerLine=true," \
#       "normalizeFractions=false," \
#       "normalizeParentheses=false," \
#       "normalizeOtherBrackets=false," \
#       "latexQuotes=false," \
#       "ptb3Ellipsis=false," \
#       "ptb3Dashes=false," \
#       "escapeForwardSlashAsterisk=false," \
#       "strictTreebank3=true' " \
#       "< ../input/{}_oneline_moses_punc.csv > ../input/{}_data.csv"
# os.system(cmd.format('train', 'train'))
# os.system(cmd.format('test', 'test'))

#5. BPE子字
subword_path = '/home/zgy/git/subword-nmt'
cmd = 'python {}/learn_joint_bpe_and_vocab.py --input ../input/train_data.csv ../input/test_data.csv ' \
      '-s 50000 -o ../input/train_test.codes --write-vocabulary ../input/train.vocab ../input/test.vocab'.format(subword_path)
os.system(cmd)
cmd = 'python {}/apply_bpe.py -c ../input/train_test.codes --vocabulary ../input/train.vocab ' \
      '--vocabulary-threshold 5 -s "" < ../input/train_data.csv > ../input/train_data_bpe.csv'.format(subword_path)
os.system(cmd)
cmd = 'python {}/apply_bpe.py -c ../input/train_test.codes --vocabulary ../input/test.vocab ' \
      '--vocabulary-threshold 5 -s ""  < ../input/test_data.csv > ../input/test_data_bpe.csv'.format(subword_path)
os.system(cmd)
cmd = 'rm ../input/train_test.codes ../input/train.vocab ../input/test.vocab'
os.system(cmd)
print('finish 5')
#6. 得到fasttext词向量

cmd='cat ../input/train_data_bpe.csv ../input/test_data_bpe.csv > ../input/content_bpe.csv && ' \
      'fasttext skipgram -input ../input/content_bpe.csv -output fasttext_bpe_300 -dim 300 -epoch 10 && ' \
      'mv fasttext_bpe_300.vec ../input/vec_fasttext_bpe_300.txt && ' \
      'mv fasttext_bpe_300.bin ../input/'
os.system(cmd)
cmd='fasttext skipgram -input ../input/content_bpe.csv -output fasttext_bpe_100 -dim 100 -epoch 10 && ' \
      'mv fasttext_bpe_100.vec ../input/vec_fasttext_bpe_100.txt ' \
      'mv fasttext_bpe_100.bin ../input/'
os.system(cmd)

cmd='cat ../input/train_data.csv ../input/test_data.csv > ../input/content.csv && ' \
      'fasttext skipgram -input ../input/content.csv -output fasttext_300 -dim 300 -epoch 10 && ' \
      'mv fasttext_300.vec ../input/vec_fasttext_300.txt ' \
      'mv fasttext_300.bin ../input/'
os.system(cmd)
cmd='fasttext skipgram -input ../input/content.csv -output fasttext_100 -dim 100 -epoch 10 && ' \
      'mv fasttext_100.vec ../input/vec_fasttext_100.txt ' \
      'mv fasttext_100.bin ../input/'
os.system(cmd)
print('finish 6')