import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from .layer_norm import LayerNorm
import ipdb

class LNGRUCell(nn.Module):
    # LayerNorm + GRU
    def __init__(self, input_size, hidden_size, bias=True, affine=True):
        super(LNGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 为了提升速度，这里将4个layer_horm合并成2个
        self.W = nn.Parameter(torch.FloatTensor(input_size, 3*hidden_size))
        self.U = nn.Parameter(torch.FloatTensor(hidden_size, 3*hidden_size))
        self.b = nn.Parameter(torch.zeros(3*hidden_size))
        self.W_ln = LayerNorm(3*hidden_size, affine=affine)
        self.U_ln = LayerNorm(3*hidden_size, affine=affine)

        self.reset()

    def reset(self):
        torch.nn.init.xavier_normal(self.W)
        torch.nn.init.xavier_normal(self.U)

    def forward(self, input, hx):
        assert input.dim()==2 and hx.dim()==2
        assert input.size(0)==hx.size(0)
        assert input.size(1)==self.input_size and hx.size(1)==self.hidden_size

        xw = torch.matmul(input, self.W)
        hu = torch.matmul(hx, self.U)
        xw1 = self.W_ln(xw)
        hu1 = self.U_ln(hu)
        xw2 = torch.split(xw1, self.hidden_size, -1)
        hu2 = torch.split(hu1, self.hidden_size, -1)

        z = F.sigmoid(xw2[0] + hu2[0] + self.b[:self.hidden_size])
        r = F.sigmoid(xw2[1] + hu2[1] + self.b[self.hidden_size:2*self.hidden_size])
        hx_ = F.tanh(r * hu2[2] + xw2[2] + self.b[2*self.hidden_size:])
        hx = (1 - z) * hx_ + z * hx
        return hx

class LNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, bias=True, dropout=0):
        super(LNGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.ff_gru = LNGRUCell(input_size, hidden_size, bias)
        self.back_gru = None
        if bidirectional:
            self.back_gru = LNGRUCell(input_size, hidden_size, bias)
        self.dropout = dropout

    def forward(self, input, h0):
        assert input.dim() == 3, input.size()
        assert h0.dim() == 3
        assert input.size(2) == self.input_size and h0.size(2) == self.hidden_size, 'input:{},h0:{}'.format(str(input.size()), str(h0.size()))
        assert input.size(1) == h0.size(1)
        assert h0.size(0) == 2 if self.bidirectional else 1
        seq_len = input.size(0)
        hiddens = []
        input_ff = [input[i] for i in range(seq_len)]
        input_back = input_ff
        hidden = h0[0] if self.bidirectional else h0
        output_ff = []
        for timestep in range(seq_len):
            x = input_ff[timestep]
            hidden = self.ff_gru(x, hidden)
            output_ff.append(F.dropout(hidden, self.dropout, self.training))
        hiddens.append(hidden)

        if self.bidirectional:
            hidden = h0[1]
            output_back = []
            for timestep in range(seq_len-1, -1, -1):
                x = input_back[timestep]
                hidden = self.back_gru(x, hidden)
                output_back.append(F.dropout(hidden, self.dropout, self.training))
            output_back.reverse()
            hiddens.append(hidden)

        hiddens = torch.stack(hiddens)
        output = torch.stack(output_ff)
        if self.bidirectional:
            output = torch.cat([output, torch.stack(output_back)], 2)
        return output, hiddens

class  cGRUCell(nn.Module):
    def __init__(self, tgt_embds_size, hidden_size, bias=True, affine=True):
        super(cGRUCell, self).__init__()
        self.tgt_embds_size = tgt_embds_size
        self.hidden_size = hidden_size

        self.rec1 = nn.GRUCell(tgt_embds_size, hidden_size)
        self.rec2 = nn.GRUCell(hidden_size, hidden_size)

        self.U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.W = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.v = nn.Parameter(torch.FloatTensor(hidden_size, 1))

    def reset_parameters(self):
        nn.init.orthogonal(self.U)
        nn.init.orthogonal(self.W)

    def attention(self, ctx, key, sentence_mask):
        '''
        :param ctx: (seqlen, batch, dim)
        :param key: (batch, dim)
        :return:
        '''
        Uk = torch.matmul(key, self.U).repeat(ctx.size(0), 1)
        Wctx = torch.matmul(ctx.view(ctx.size(0)*ctx.size(1), ctx.size(2)), self.W)
        e = torch.matmul(F.tanh(Uk + Wctx), self.v)
        e = F.softmax(torch.cat(torch.split(e, ctx.size(1), 0), 1)+sentence_mask)
        attn = torch.bmm(e.unsqueeze(1), ctx.permute(1,0,2))
        return attn[:,0,:]

    def forward(self, hx, y, ctx, sentence_mask):
        _hx = self.rec1(y, hx)
        c = self.attention(ctx, _hx, sentence_mask)
        hx = self.rec2(c, _hx)
        return hx

class  cLSTMCell(nn.Module):
    def __init__(self, embds_size, hidden_size, ctx_size):
        super(cLSTMCell, self).__init__()
        self.embds_size = embds_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size

        self.rec1 = nn.LSTMCell(embds_size, hidden_size)
        self.rec2 = nn.LSTMCell(hidden_size + ctx_size, hidden_size)

    def forward(self, y_embeds, pre_hiddens, ctx, low_inputs):
        '''
        :param y_embeds: (batch, embeds_size)
        :param pre_hiddens: ((batch, hidden_size),(batch, hidden_size))
        :param ctx:(batch, ctx_size)
        :param low_inputs:(batch, hidden_size)
        :return:((batch, hidden_size),(batch, hidden_size))
        '''
        hx, cx = self.rec1(y_embeds, pre_hiddens)
        hx, cx = self.rec2(torch.cat([ctx, low_inputs], -1), (hx, cx))
        return (hx, cx)

if __name__=='__main__':
    #hx, y, ctx, sentence_mask
    rnn = cLSTMCell(3, 4)
    hx = Variable(torch.randn(2, 4))
    y = Variable(torch.randn(2, 3))
    ctx = Variable(torch.randn(3, 2, 4))
    sentence_mask = Variable(torch.randn(2, 3))
    hx = rnn(hx, y, ctx, sentence_mask)
    print(hx)








