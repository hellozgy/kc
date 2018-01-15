# coding: utf-8

import pandas as pd
import re, string, os
import ipdb

# 1.将数据标签分开，并将内容转化为一行
train = '../input/train.csv'
test = '../input/test.csv'
train_oneline = '../input/train_oneline.csv'
train_label = '../input/train_label.csv'
test_oneline = '../input/test_oneline.csv'
test_label = '../input/test_label.csv'

train = pd.read_csv(train)
test = pd.read_csv(test)
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

train[COMMENT].str.replace('\n','').to_csv(train_oneline, header=False, index=False, encoding='utf-8')
del train[COMMENT]
train.to_csv(train_label, header=False, index=False, encoding='utf-8')
test[COMMENT].str.replace('\n', '').to_csv(test_oneline, header=False, index=False, encoding='utf-8')
del test[COMMENT]
test.to_csv(test_label, header=False, index=False, encoding='utf-8')

#2.去除不可见字符，将中文标点转化为英文标点
# moses_dir = 'd:/git/mosesdecoder-master/scripts/tokenizer'
moses_dir = '/home/zgy/git/mosesdecoder-master/scripts/tokenizer'
cmd = 'cat ../input/train_oneline.csv | perl {}/remove-non-printing-char.perl | perl {}/replace-unicode-punctuation.perl > ' \
      '../input/train_oneline_moses.csv'.format(moses_dir, moses_dir)
os.system(cmd)
cmd = 'cat ../input/test_oneline.csv | perl {}/remove-non-printing-char.perl | perl {}/replace-unicode-punctuation.perl > ' \
      '../input/test_oneline_moses.csv'.format(moses_dir, moses_dir)
os.system(cmd)

#3. 删除首尾标点和重复标点
train_data_src = '../input/train_oneline_moses.csv'
train_data_dst = '../input/train_oneline_moses_punc.csv'
test_data_src = '../input/test_oneline_moses.csv'
test_data_dst = '../input/test_oneline_moses_punc.csv'
def filter(src, dst):
    f = open(src, 'r', encoding='utf-8')
    fw = open(dst, 'w', encoding='utf-8')
    for line in f.readlines():
        line = line.strip().lower()
        line = re.sub(r'\d', '', line)
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
        line = re.sub(fr'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’ \n])+', r'\1', line)
        line = re.sub(r'([!"#$%&\()*+,-./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘])\1*', r' \1 ', line)
        line = re.sub('\\s+', ' ', line)
        fw.write(line.strip()+'\n')
        fw.flush()
    f.close()
    fw.close()
            
filter(train_data_src, train_data_dst)
filter(test_data_src, test_data_dst)

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
os.system(cmd.format('train', 'train'))
os.system(cmd.format('test', 'test'))

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









