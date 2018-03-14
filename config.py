import os

class Config():
    ngpu = -1
    model = 'LNGRUText'
    id = None
    seed = 1

    max_len = 200
    embeds_size = 100
    hidden_size = 128
    dropout = 0.2
    log_iter = 100
    batch_size = 64
    epochs = 30
    lr = 1e-3
    limit_lr = 1e-6
    restore = False
    restore_file = None
    vocab_size = -1
    num_classes = 6
    save_model = True
    docs_file = 'docs_bpe.npz'
    embeds_path = './input/vec_fasttext_bpe_100.txt'
    split_sentence = False
    subset = 'commit'
    index = 7
    eval_every = 2000
    models=None


    filters = [2, 3, 4, 5]
    filter_nums = [128, 128, 256, 256]
    model_type = 'multichannel'
    tune = False
    num_layers = 1
    kmax_pooling = 1
    linear_hidden_size = 512
    weight_decay = 1e-5
    dropout_data = 0.3
    res_file = 'res.csv'
    bpe=True

    def parse(self, args):
        for k, v in args.items():
            assert hasattr(self, k), 'opt has no attribute:'+str(k)
            setattr(self, k, v)
        return self

    def show(self):
        keys = [k for k in dir(self) if not k.startswith('_') and not k.startswith('parse') and not k.startswith('show')]
        for key in keys:
            print("{}:{}".format(key, getattr(self, key)))

    def parseopt(self, opt):
        keys = [k for k in dir(opt) if not k.startswith('_') and not k.startswith('parse')
                and not k.startswith('show') and k not in set(['ngpu', 'id', 'restore_file', 'res_file'])]
        for key in keys:
            setattr(self, key, getattr(opt, key))

opt = Config()
