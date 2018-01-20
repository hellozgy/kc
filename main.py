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
    opt.id = opt.model
    assert opt.ngpu >= 0
    train_data = KCDataset('docs_bpe.npz', ['train', 'val'], opt.max_len)
    test_data = KCDataset('docs_bpe.npz', ['test'], opt.max_len)
    opt.vocab_size = train_data.vocab_size
    model = getattr(models, opt.model)(opt)
    restore_file = './checkpoints/{}/{}'.format(opt.model,
                                                'checkpoint_last' if opt.restore_file is None else opt.restore_file)
    save2path = './checkpoints/{}/'.format(opt.model)
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

    optimizer = model.get_optimizer(opt.lr)
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
            if batch % opt.log_iter == 0:
                print('epoch:{}-batch:{}-loss:{}'.format(epoch, batch, loss / opt.log_iter))
                loss = 0

        min_loss, checkpoint_id = test(test_data, opt, model, min_loss, checkpoint_id, epoch)

def test(dataset, opt, model, min_loss, checkpoint_id, epoch):
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
    print('eval:epoch:{}-loss:{}'.format(epoch, loss))

    if opt.save_model:
        torch.save({'model': model.state_dict(), 'checkpoint_id': checkpoint_id, 'loss': loss, 'opt': opt},
                   './checkpoints/{}/checkpoint{}'.format(opt.id, checkpoint_id))
        shutil.copy('./checkpoints/{}/checkpoint{}'.format(opt.id, checkpoint_id),
                    './checkpoints/{}/checkpoint_last'.format(opt.id))
        if loss < min_loss:
            min_loss = loss
            shutil.copy('./checkpoints/{}/checkpoint_last'.format(opt.id),
                        './checkpoints/{}/checkpoint_best'.format(opt.id))
    return min_loss, checkpoint_id+1


if __name__ == '__main__':  
    import fire
    fire.Fire()
    '''
    python main.py train --ngpu=9
    '''