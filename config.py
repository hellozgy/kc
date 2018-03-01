import os

class Config():
    ngpu = -1  # 指定gpu
    model = 'BasicModule'
    id = None

    max_len = 100
    embeds_size = 300
    hidden_size = 256
    dropout = 0.5
    log_iter = 100
    batch_size = 64
    epochs = 30
    lr = 1e-3
    limit_lr = 1e-6
    restore = True
    restore_file = None
    vocab_size = -1
    num_classes = 6
    save_model = False
    docs_file = 'docs_bpe.npz'
    embeds_path = './input/vec_glove_bpe_300.txt'
    split_sentence = False
    subset = 'commit'  # test函数的数据
    index = 7

    # CNNText 参数
    filters = [2, 3, 4, 5]
    filter_nums = [128, 128, 256, 256]
    model_type = 'multichannel'
    tune = False # 使用测试集微调
    num_layers = 1
    kmax_pooling = 3
    linear_hidden_size = 2000
    weight_decay = 2e-5
    dropout_data = 0.7

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
                and not k.startswith('show') and k not in set(['ngpu', 'id', 'restore_file'])]
        for key in keys:
            setattr(self, key, getattr(opt, key))

opt = Config()
