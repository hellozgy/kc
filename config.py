
class Config():
    ngpu = -1  # 指定gpu
    model = None
    max_len = 30
    dropout = 0.
    eval_iter = 1000  # 评估模型
    batch_size = 128
    epochs = 10
    lr = 1e-3
    limit_lr = 1e-6
    restore_file = None

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
                and not k.startswith('show') and k not in set(['ngpu', 'id', 'beam_size', 'alpha', 'beta', 'restore_file'])]
        for key in keys:
            setattr(self, key, getattr(opt, key))


opt = Config()
