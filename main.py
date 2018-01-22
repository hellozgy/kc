#coding:utf-8
import os
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
from config import opt
from dataset import KCDataset
import models
import torch
import shutil
import ipdb

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def train(**kwargs):
    opt.parse(kwargs)
    opt.id = opt.model if opt.id is None else opt.id
    assert opt.ngpu >= 0
    train_data = KCDataset('docs_bpe.npz', ['train', 'val'], opt.max_len, split_sentence=opt.split_sentence)
    test_data = KCDataset('docs_bpe.npz', ['test'], opt.max_len, split_sentence=opt.split_sentence)
    opt.vocab_size = train_data.vocab_size
    model = getattr(models, opt.model)(opt)
    restore_file = './checkpoints/{}/{}'.format(opt.id,
                                                'checkpoint_last' if opt.restore_file is None else opt.restore_file)
    save2path = './checkpoints/{}/'.format(opt.id)
    if not os.path.exists(save2path): os.system('mkdir -p {}'.format(save2path))
    min_loss = float('inf')
    checkpoint_id = 1
    if os.path.exists(restore_file):
        print('restore parameters from {}'.format(restore_file))
        model_file = torch.load(restore_file)
        opt.parseopt(model_file['opt'])
        model = getattr(models, opt.model)(opt)
        model.load_state_dict(model_file['model'], strict=False)
        checkpoint_id = int(model_file['checkpoint_id']) + 1
        min_loss = float(model_file['loss'])
    model.cuda(opt.ngpu)

    optimizer = model.get_optimizer(opt.lr, lr2=0, weight_decay=2e-5)
    dataloader_train = data.DataLoader(
        dataset=train_data, batch_size=opt.batch_size,
        shuffle=True, num_workers=1, drop_last=False)
    loss_function = nn.BCELoss(size_average=True)
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
        print('epoch:{}-loss:{:,.5f}'.format(epoch, loss / batch))

        epoch_loss, checkpoint_id = eval(test_data, opt, model, min_loss, checkpoint_id, epoch)
        if epoch_loss <= min_loss:
            min_loss = epoch_loss
        else:
            opt.lr = opt.lr / 2
            optimizer = model.get_optimizer(opt.lr, lr2=opt.lr / 2, weight_decay=2e-5)

def eval(dataset, opt, model, min_loss, checkpoint_id, epoch):
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
    msg = 'eval:epoch:{}-loss:{:,.5f}-minloss:{:,.5f}'.format(epoch, loss, min_loss)
    print(msg)
    os.system('echo {} >> ./output/{}.txt'.format(msg, opt.id))

    if opt.save_model:
        torch.save({'model': model.state_dict(), 'checkpoint_id': checkpoint_id, 'loss': loss, 'opt': opt},
                   './checkpoints/{}/checkpoint_last'.format(opt.id))
        if loss <= min_loss:
            min_loss = loss
            shutil.copy('./checkpoints/{}/checkpoint_last'.format(opt.id),
                        './checkpoints/{}/checkpoint_best'.format(opt.id))
    return min_loss, checkpoint_id+1

def test(**kwargs):
    opt.parse(kwargs)
    opt.id = opt.model if opt.id is None else opt.id
    assert opt.ngpu >= 0
    test_data = KCDataset('docs_bpe.npz', ['commit'], opt.max_len, split_sentence=opt.split_sentence)
    opt.vocab_size = test_data.vocab_size
    model = getattr(models, opt.model)(opt)
    restore_file = './checkpoints/{}/{}'.format(opt.id,
                                                'checkpoint_best' if opt.restore_file is None else opt.restore_file)
    if os.path.exists(restore_file):
        print('restore parameters from {}'.format(restore_file))
        model_file = torch.load(restore_file)
        opt.parseopt(model_file['opt'])
        model = getattr(models, opt.model)(opt)
        model.load_state_dict(model_file['model'], strict=False)
    model.cuda(opt.ngpu)

    dataloader_train = data.DataLoader(
        dataset=test_data, batch_size=opt.batch_size,
        shuffle=False, num_workers=1, drop_last=False)
    res_file = './output/res_{}.csv'.format(opt.id)
    fw = open(res_file, 'w', encoding='utf-8')
    fw.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
    for content, ids in dataloader_train:
        content = Variable(content, volatile=True).long().cuda(opt.ngpu)
        predicts = model(content).cpu().data.numpy().tolist()
        res = ''
        for id, predict in zip(ids, predicts):
            res += '{},{}\n'.format(id, ','.join(list(map(str, predict))))
        fw.write(res)
        fw.flush()
    fw.close()


if __name__ == '__main__':  
    import fire
    fire.Fire()
    '''
    python main.py train --ngpu=9
    '''