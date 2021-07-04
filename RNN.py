'''
Author: your name
Date: 2021-04-28 10:36:10
LastEditTime: 2021-05-12 11:41:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /dl_2021_hw3/dl_2021_hw3/RNN.py
'''
'''
This code is based on example code in DSGA-1008-Spring2017-A2, NYU // 
https://github.com/RMichaelSwan/MogrifierLSTM/blob/master/MogrifierLSTM.ipynb

'''

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, bidirectional=False,
        tie_weights=False, xavier=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.encoder = nn.Embedding(ntoken, ninp)
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout,bidirectional=bidirectional,bias=False)
        elif rnn_type == 'MogLSTM':
            self.rnn = MogLSTM(ninp, nhid, 5)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                                                 dropout=dropout,bidirectional=bidirectional,bias=False)
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        
        self.decoder = nn.Linear(nhid, ntoken)
        self.xavier = xavier

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def _calculate_fan_in_and_fan_out(self, tensor):
        if tensor.ndimension() < 2:
            raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

        if tensor.ndimension() == 2: 
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = np.prod(tensor.numpy().shape[2:])
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def xavier_normal(self, tensor, gain=1):
        ## tensor here refers to weight tensor to be initialized
        if isinstance(tensor, Variable):
            self.xavier_normal(tensor.data, gain=gain)
            return tensor
        else:
            fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
            std = gain * np.sqrt(2.0 / (fan_in + fan_out))
            return tensor.normal_(0, std)
        
    def init_weights(self):
        initrange = 0.1
        if self.xavier:
            self.xavier_normal(self.encoder.weight, gain=np.sqrt(2))
            self.xavier_normal(self.decoder.weight, gain=np.sqrt(2))
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = self.drop(decoded)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


## 
class MogLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, mog_iterations: int):
        super(MogLSTM, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.mog_iterations = mog_iterations
        #Define/initialize all tensors   
        self.Wih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))
        #Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_sz,input_sz))
        self.R = Parameter(torch.Tensor(input_sz,hidden_sz))

        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self,xt,ht):
        for i in range(1,self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2*torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht

    
    def forward(self, x, init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        seq_sz, batch_sz, _ = x.size()
        hidden_seq = []
        #ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz): # iterate over the time steps
            xt = x[t, :, :]
            xt, ht = self.mogrify(xt, ht) #mogrification
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ### The LSTM Cell!
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            #outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            ###

            hidden_seq.append(ht.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (ht, Ct)
