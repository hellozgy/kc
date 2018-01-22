from .BasicModule import BasicModule
from torch import nn
import torch
import numpy as np
from dataset.Constants import PAD_INDEX
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

class HAN(BasicModule):
    def __init__(self, opt, word_layers=2, sent_layers=2):
        super(HAN, self).__init__(opt)
        self.vocab_size = opt.vocab_size
        assert self.vocab_size > 0
        self.embeds_size = opt.embeds_size
        self.hidden_size = opt.hidden_size
        self.dropout = opt.dropout
        self.num_classes = opt.num_classes
        self.embeds = nn.Embedding(self.vocab_size, self.embeds_size, padding_idx=PAD_INDEX)
        if opt.embeds_path:
            self.embeds.weight.data.copy_(torch.from_numpy(np.load(opt.embeds_path)['vector']))

        self.word_encoder = nn.LSTM(self.embeds_size, self.hidden_size, bidirectional=True, num_layers=word_layers)
        self.sent_encoder = nn.LSTM(2*self.hidden_size, self.hidden_size, bidirectional=True, num_layers=sent_layers)
        self.fc_word = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.fc_sent = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.query_word = nn.Parameter(torch.FloatTensor(2*self.hidden_size, 1))
        self.query_sent = nn.Parameter(torch.FloatTensor(2*self.hidden_size, 1))
        nn.init

        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, content):
        '''
        :param content: batch * max_sents * max_word
        :return:
        '''
        batch, max_sents, max_word = content.size()
        words = content.view(batch * max_sents, max_word)
        mask = torch.eq(words, PAD_INDEX).data
        mask_sent = torch.eq(torch.sum(mask, dim=1), max_word)
        mask = mask.float().masked_fill_(mask, -float('inf'))
        mask = mask.index_fill_(0, mask_sent.float().topk(int(torch.sum(mask_sent.float())))[1], 0)
        mask_sent = mask_sent.float().masked_fill_(mask_sent, -float('inf')).view(batch, max_sents)
        sents, _ = self.word_encoder(self.embeds(words).permute(1, 0, 2))
        sents, _ = self.attention(sents, self.fc_word, self.query_word, Variable(mask))
        sents = sents.view(batch, max_sents, 2*self.hidden_size).permute(1, 0, 2)
        sents, _ = self.sent_encoder(sents)
        sents, _ = self.attention(sents, self.fc_sent, self.query_sent, Variable(mask_sent))
        output = self.fc(sents)
        return output

    def attention(self, ctx, Ww, Uw, mask):
        ctx = ctx.permute(1, 0, 2)
        u = F.tanh(Ww(ctx))
        score = F.softmax(u.matmul(Uw).squeeze(2)+mask, dim=1).unsqueeze(1)
        attn = torch.bmm(score, ctx)[:, 0, :]  # (batch_size,1, hidden_size)
        return attn, score[:, 0, :]


