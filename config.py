import os

base_dir = os.path.abspath(os.path.dirname(__file__) + '/input')


class Config():
    ngpu = 0  # 指定gpu
    # model = 'FastText'
    # model = 'CNNText'
    model = 'CNNInceptionText'
    id = None
    max_len = 100
    embeds_size = 300
    hidden_size = 256
    dropout = 0.5
    log_iter = 100
    batch_size = 64
    epochs = 10
    lr = 1e-3
    limit_lr = 1e-6
    restore = False
    restore_file = None
    vocab_size = -1
    num_classes = 6
    inception_dim = 512
    save_model = True
    embeds_path = os.path.join(base_dir, 'vec_fasttext_bpe.npz')
    split_sentence = False

    # CNNText 参数
    filters = [3, 5, 7]
    filter_nums = [100, 100, 100]
    model_type = 'multichannel'


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
