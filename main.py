#coding:utf-8
import os
import numpy as np
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
from config import opt
from dataset import KCDataset, KCDataset10fold, KCDatasetTest
import models
import torch
import shutil
import datetime
import numpy as np
from meter import MulLabelConfusionMeter
from loss_module import BCELossWeight
import ipdb

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

data_dir = os.path.abspath(os.path.dirname(__file__) + './input/')

def train(**kwargs):
    torch.cuda.set_device(opt.ngpu)
    opt.parse(kwargs)
    opt.id = opt.model if opt.id is None else opt.id
    assert opt.ngpu >= 0
    # train_data = KCDataset(opt.docs_file, ['train', 'val'], opt.max_len, split_sentence=opt.split_sentence, training=True, dropout_data=opt.dropout_data)
    # test_data = KCDataset(opt.docs_file, ['test'], opt.max_len, split_sentence=opt.split_sentence, training=False)
    print('load data...')
    dataset = KCDataset10fold(index=opt.index, max_len=opt.max_len, vec_name=opt.embeds_path, training=True, dropout_data=0.3)
    opt.vocab_size = dataset.vocab_size
    print('load model...')
    model = getattr(models, opt.model)(opt)
    restore_file = './checkpoints/{}/{}'.format(opt.id, opt.restore_file)
    save2path = './checkpoints/{}/'.format(opt.id)
    if not os.path.exists(save2path): os.system('mkdir -p {}'.format(save2path))
    min_loss = float('inf')
    checkpoint_id = 1
    if  opt.restore and os.path.exists(restore_file):
        print('restore parameters from {}'.format(restore_file))
        model_file = torch.load(restore_file)
        opt.parseopt(model_file['opt'])
        model = getattr(models, opt.model)(opt)
        model.load_state_dict(model_file['model'], strict=False)
        checkpoint_id = int(model_file['checkpoint_id']) + 1
        min_loss = float(model_file['loss'])
    model.cuda(opt.ngpu)


    lr2 = 0
    optimizer = model.get_optimizer(opt.lr if not opt.tune else 1e-5, lr2=lr2 if not opt.tune else 1e-5, weight_decay=opt.weight_decay)
    dataloader_train = data.DataLoader(
        dataset=dataset, batch_size=opt.batch_size,
        shuffle=True, num_workers=1, drop_last=False)
    loss_function = nn.BCELoss(size_average=True)
    fw = open('./checkpoints/{}/log.txt'.format(opt.id), 'a', encoding='utf-8')
    # tag = 3
    print('train...')
    for epoch in range(opt.epochs):
        loss = 0
        batch = 0
        for content, label in dataloader_train:
            content = Variable(content).long().cuda(opt.ngpu)
            label = Variable(label).float().cuda(opt.ngpu)
            batch += 1
            optimizer.zero_grad()

            predict = model(content)
            batch_loss = loss_function(predict, label)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.data[0]

        epoch_loss, checkpoint_id = eval(dataset, opt, model, min_loss, checkpoint_id)
        min_loss = min(min_loss, epoch_loss)

        msg = '{} epoch:{:>2} train_loss:{:,.6f} test_loss:{:,.6f} minloss:{:,.6f} lr:{:,.6f}'.format(
            str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), epoch, loss / batch, epoch_loss, min_loss,opt.lr)
        print(msg)
        fw.write(msg+'\n')
        fw.flush()

        if epoch_loss > min_loss:
            # tag -= 1
            # if tag==0:
            #     tag=2
            opt.lr = opt.lr * 0.5
            lr2 = 2e-4 if lr2==0 else lr2*0.5
            model.load_state_dict(torch.load('./checkpoints/{}/checkpoint_best'.format(opt.id))['model'])
            optimizer = model.get_optimizer(opt.lr if not opt.tune else 1e-5, lr2=lr2 if not opt.tune else 1e-5, weight_decay=opt.weight_decay)
            # model.update_optimizer(optimizer, opt.lr, lr2)

    fw.close()

def eval(dataset, opt, model, min_loss, checkpoint_id):
    print('eval...')
    dataset.set_train(train=False)
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=1, drop_last=False)
    loss = 0
    step = 0
    model.eval()
    loss_function = nn.BCELoss(size_average=True)
    for content, label in dataloader:
        step += 1
        content = Variable(content, volatile=True).long().cuda(opt.ngpu)
        label = Variable(label, volatile=True).float().cuda(opt.ngpu)
        predict = model(content)
        batch_loss = loss_function(predict, label)
        loss += batch_loss.data[0]
    model.train()
    loss = loss / step

    if opt.save_model:
        torch.save({'model': model.state_dict(), 'checkpoint_id': checkpoint_id, 'loss': loss, 'opt': opt},
                   './checkpoints/{}/checkpoint_last'.format(opt.id))
        if loss <= min_loss:
            shutil.copy('./checkpoints/{}/checkpoint_last'.format(opt.id),
                        './checkpoints/{}/checkpoint_best'.format(opt.id))
    dataset.set_train(train=True)
    return loss, checkpoint_id+1

def test(**kwargs):
    opt.parse(kwargs)
    opt.id = opt.model if opt.id is None else opt.id
    assert opt.ngpu >= 0
    # assert ('glove' in opt.docs_file and 'glove' in opt.embeds_path) or ('glove' not in opt.docs_file and 'glove' not in opt.embeds_path)
    test_data = KCDatasetTest('./tenfold/train_data_test_7.csv', '.tenfold/train_label_test_7.csv')
    opt.vocab_size = test_data.vocab_size
    restore_file = './checkpoints/{}/{}'.format(opt.id,
                                                'checkpoint_best' if opt.restore_file is None else opt.restore_file)
    print('restore parameters from {}'.format(restore_file))
    model_file = torch.load(restore_file)
    opt.parseopt(model_file['opt'])
    model = getattr(models, opt.model)(opt)
    model.load_state_dict(model_file['model'], strict=False)
    model.cuda(opt.ngpu)

    dataloader= data.DataLoader(
        dataset=test_data, batch_size=opt.batch_size,
        shuffle=False, num_workers=1, drop_last=False)

    res_file = './checkpoints/{}/res.csv'.format(opt.id)
    fw = open(res_file, 'w', encoding='utf-8')
    fw.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
    loss_function = nn.BCELoss(size_average=True)
    # loss_function = BCELossWeight(opt.ngpu)
    # loss_function.reset()
    confusion_matrix = MulLabelConfusionMeter(num_class=6)
    loss = 0
    step = 0
    model.eval()
    for content, labels, ids in dataloader:
        step += 1
        content = Variable(content, volatile=True).long().cuda(opt.ngpu)
        labels = Variable(labels, volatile=True).float().cuda(opt.ngpu)
        predicts = model(content)

        batch_loss = loss_function(predicts, labels)
        loss += batch_loss.data[0]
        confusion_matrix.add(predicts.cpu().data.numpy(), labels.cpu().data.numpy(), opt.batch_size, step-1)

        res = ''
        for id, p in zip(ids, predicts.cpu().data.numpy()):
            res += '{},{}\n'.format(id, ','.join(list(map(str, p))))
        fw.write(res)
        fw.flush()
    fw.close()

    with open('./checkpoints/{}/log_test.txt'.format(opt.id), 'a', encoding='utf-8') as flog:
        print('loss:{:,.6f}\n'.format(loss/step))
        flog.write('---------------------loss:{:,.5f}---------------------\n'.format(loss/step))
        flog.write(str(confusion_matrix)+'\n')

if __name__ == '__main__':  
    import fire
    fire.Fire()
    '''
    python main.py train --ngpu=9
    python main.py train --ngpu=9 --model=LSTMText --save-model=True --restore=False --hidden-size=128 --epochs=20 --batch_size=64 --num_layers=4 \
    --kmax-pooling=1 --linear-hidden-size=512 --weight-decay=1e-4 --dropout-data=0.3 --id=LSTMText_h128
    '''