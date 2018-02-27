# coding: utf-8

import pandas as pd
import re, string, os
import unicodedata
import ipdb

# 1.将数据标签分开，并将内容转化为一行
outdata_oneline = '../input/outdata_oneline.csv'

#2.去除不可见字符，将中文标点转化为英文标点
# moses_dir = 'd:/git/mosesdecoder-master/scripts/tokenizer'
moses_dir = '/home/zgy/git/mosesdecoder-master/scripts/tokenizer'
cmd = 'cat ../input/outdata_oneline.csv | perl {}/remove-non-printing-char.perl | perl {}/replace-unicode-punctuation.perl > ' \
      '../input/outdata_oneline_moses.csv'.format(moses_dir, moses_dir)
os.system(cmd)

#3. 删除首尾标点和重复标点
outdata_data_src = '../input/outdata_oneline_moses.csv'
outdata_data_dst = '../input/outdata_oneline_moses_punc.csv'
def filter(src, dst):
    f = open(src, 'r', encoding='utf-8')
    fw = open(dst, 'w', encoding='utf-8')
    for line in f.readlines():
        line = line.strip().lower()
        line = unicodedata.normalize('NFKC', line)
        line = re.sub(r'\d', '', line)
        line = re.sub(rf'([^{string.punctuation}a-zA-Z])', ' ', line) # 仅保留标点和字母
        line = re.sub(r'[#$%+=~^|@_/]', ' ', line) # '#'作为句子之间的分隔符
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
        line = re.sub(fr'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’ \n])\1*', r'\1', line)
        # line = re.sub(r'([!"#$%&\()*+,-./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘])\1*', r' \1 ', line)
        line = re.sub(r'([^*`\'a-zA-Z])\1*', r' \1 ', line)
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
        line = line.strip()
        fw.write((line if len(line)>0 else 'unknown')+'\n')
        fw.flush()
    f.close()
    fw.close()
            
filter(outdata_data_src, outdata_data_dst)

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
subword_path = '/home/zgy/git/subword-nmt'
cmd = 'python {}/learn_joint_bpe_and_vocab.py --input ../input/train_data.csv ../input/test_data.csv ' \
      '-s 50000 -o ../input/train_test.codes --write-vocabulary ../input/train.vocab ../input/test.vocab'.format(subword_path)
os.system(cmd)
cmd = 'python {}/apply_bpe.py -c ../input/train_test.codes --vocabulary ../input/train.vocab --vocabulary-threshold 5 -s "" ' \
      ' < ../input/outdata_data.csv > ../input/outdata_data_bpe.csv'.format(subword_path)
os.system(cmd)

