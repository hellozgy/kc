
def generate_word(fdoc, fword):
    '''生成单词表'''
    res = {}
    with open(fdoc, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                res[word] = res.get(word, 0)+1
    res = [(k,v) for k,v in res.items()]
    res.sort(key=lambda x:x[1], reverse=True)

    with open(fword, 'w', encoding='utf-8') as fw:
        fw.write('<pad> 1000000\n')
        fw.write('<unk> 1000000\n')
        for (k,v) in res:
            fw.write(k+' '+str(v)+'\n')

generate_word('../input/content_bpe.csv', '../input/word_list_bpe.csv')
generate_word('../input/content.csv', '../input/word_list.csv')
