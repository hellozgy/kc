from .BasicModule import BasicModule
from torch import nn
import torch
import numpy as np
from dataset.Constants import PAD_INDEX
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

class HAN(BasicModule):
    def __init__(self, opt):
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

        self.word_encoder = nn.LSTM(self.embeds_size, self.hidden_size, bidirectional=True, num_layers=opt.num_layers)
        self.sent_encoder = nn.LSTM(6*self.hidden_size, self.hidden_size, bidirectional=True, num_layers=opt.num_layers)
        self.fc_word = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.fc_sent = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.query_word = nn.Parameter(torch.FloatTensor(2*self.hidden_size, 1))
        self.query_sent = nn.Parameter(torch.FloatTensor(2*self.hidden_size, 1))
        self.attn_Wa_word = nn.Parameter(torch.FloatTensor(4*self.hidden_size, 2*self.hidden_size))
        self.attn_Va_word = nn.Parameter(torch.FloatTensor(2*self.hidden_size, 1))
        self.attn_fc_word = nn.Linear(4*self.hidden_size, 2*self.hidden_size)
        self.attn_Wa_sent = nn.Parameter(torch.FloatTensor(4 * self.hidden_size, 2 * self.hidden_size))
        self.attn_Va_sent = nn.Parameter(torch.FloatTensor(2 * self.hidden_size, 1))
        self.attn_fc_sent = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size)

        nn.init.xavier_normal(self.query_word)
        nn.init.xavier_normal(self.query_sent)
        nn.init.xavier_normal(self.attn_Wa_word)
        nn.init.xavier_normal(self.attn_Va_word)
        nn.init.xavier_normal(self.attn_Wa_sent)
        nn.init.xavier_normal(self.attn_Va_sent)

        self.fc = nn.Sequential(
            nn.Linear(6 * self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, content, lengths):
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

        words_embeds = self.embeds(words).permute(1, 0, 2)
        words_embeds = F.dropout(words_embeds, p=self.dropout, training=self.training)
        sents, (ht,_) = self.word_encoder(words_embeds)
        ht = torch.cat([ht[-2], ht[-1]], dim=1) # (batch, 2*hidden_size)
        sents_attn, _ = self.attention_han(sents, self.fc_word, self.query_word, Variable(mask))
        sents_attn2, _ = self.attention(sents, ht, self.attn_Wa_word, self.attn_Va_word, self.attn_fc_word,  Variable(mask))
        sents = torch.cat([ht, sents_attn, sents_attn2], -1).view(batch, max_sents, 6*self.hidden_size).permute(1, 0, 2)

        sents, (ht, _) = self.sent_encoder(sents)
        ht = torch.cat([ht[-2], ht[-1]], dim=1)
        sents_attn, _ = self.attention_han(sents, self.fc_sent, self.query_sent, Variable(mask_sent))
        sents_attn2, _ = self.attention(sents, ht, self.attn_Wa_sent, self.attn_Va_sent, self.attn_fc_sent,  Variable(mask_sent))
        sents = torch.cat([ht, sents_attn, sents_attn2], -1)

        output = self.fc(sents)
        return output

    def attention_han(self, ctx, Ww, Uw, mask):
        ctx = ctx.permute(1, 0, 2)
        u = F.tanh(Ww(ctx))
        score = F.softmax(u.matmul(Uw).squeeze(2)+mask, dim=1).unsqueeze(1)
        attn = torch.bmm(score, ctx)[:, 0, :]  # (batch_size,1, hidden_size)
        return attn, score[:, 0, :]

    def attention(self, ctx, key, attn_Wa, attn_Va, attn_fc,  mask):
        '''
        :param ctx: (seq_len, batch_size, hidden_size)
        :param key: (batch_size, hidden_size)
        :param mask: (batch_size, seq_len)
        :return:
        :attn:(batch_size, hidden_size)
        :At:(batch_size, seq_len)
        '''
        residual = key
        seq_len = ctx.size(0)
        ctx = ctx.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size*n)
        key = torch.stack([key]*seq_len, 1) #(batch_size, seq_len, hidden_size)
        At = F.tanh(torch.cat([ctx, key], -1).matmul(attn_Wa)).matmul(attn_Va).squeeze(-1)
        At = At + mask
        At = F.softmax(At, dim=1).unsqueeze(1)  # (batch_size,1, seqlen)
        attn = torch.bmm(At, ctx)[:,0,:]  # (batch_size,1, hidden_size)
        attn = F.tanh(attn_fc(torch.cat([attn, residual], 1)))
        return attn, At[:, 0, :]


