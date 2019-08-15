import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SimpleLSTM(nn.Module):
    def __init__(self, nIn, nHidden):
        super(SimpleLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=False)

    def forward(self, input_):
        recurrent, hidden = self.rnn(input_)
        return recurrent


class SimpleLinear(nn.Module):
    def __init__(self, nIn, nOut):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(nIn, nOut)

    def forward(self, x):
        timesteps, batch_size = x.size(0), x.size(1)
        x = x.view(timesteps*batch_size, -1)
        x = self.linear(x)
        # x = x.view(timesteps, batch_size, -1)
        # x = x.unsqueeze(1)
        return x



class Delayed_LSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, bidirectional=False):
        super(Delayed_LSTM, self).__init__()
        self.nHidden = nHidden
        self.embedding = nn.Embedding(nIn, nHidden)
        self.n_layers = 1
        self.bi = 1
        if bidirectional == True:
            self.bi = 2
        self.lstm = nn.LSTM(nHidden, nHidden, bidirectional=bidirectional)
        # self.lstm = nn.Sequential(*self.lstm)
        self.fc_out = nn.Linear(self.bi*nHidden, nOut)

    def initHidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.bi*self.n_layers, batch_size, self.nHidden).zero_().cuda(),
                  weight.new(self.bi*self.n_layers, batch_size, self.nHidden).zero_().cuda())
        else:
            hidden = (weight.new(self.bi*self.n_layers, batch_size, self.nHidden).zero_(),
                      weight.new(self.bi*self.n_layers, batch_size, self.nHidden).zero_())
        
        return hidden

    def forward(self, x, hidden):
        x = self.embedding(x) # BTH
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2) #TBH
        x, hidden = self.lstm(x, hidden)
        x = x.permute(1, 0, 2) #BTH
        B, T, H = x.size()
        x = x.contiguous().view(-1, H)
        out = self.fc_out(x)
        return out, hidden


class EncoderRNN(nn.Module):
    def __init__(self, nClasses, embed_dim, enc_units):
        super(EncoderRNN, self).__init__()
        self.input_size = nClasses
        self.enc_units = enc_units
        self.embedding = nn.Embedding(nClasses, embed_dim)
        self.bi = False
        self.rnn = nn.GRU(
            embed_dim, enc_units,
            dropout = 0.5,
            bidirectional = self.bi,
            batch_first = True)

    def forward(self, inputs, input_lengths):
        x = self.embedding(inputs)
        batch_size = x.size(0)
        x = pack_padded_sequence(x, input_lengths, batch_first=True)
        self.hidden = self.init_hidden(batch_size)
        output, self.hidden = self.rnn(x, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        if self.bi:
            output = output[:, :, :self.enc_units] + output[:, :, self.enc_units:]
        return output, self.hidden

    def init_hidden(self, batch_size):
        if self.bi:
            return torch.zeros((2, batch_size, self.enc_units)).cuda()
        return torch.zeros((1, batch_size, self.enc_units)).cuda()

class RNN_Decoder(nn.Module):
    def __init__(self, nClasses, embed_dim, enc_units, dec_units):
        super(RNN_Decoder, self).__init__()
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.embedding = nn.Embedding(nClasses, embed_dim)
        self.bi = False
        self.rnn = nn.GRU(
            embed_dim + enc_units,
            dec_units,
            dropout=0.5,
            bidirectional=self.bi,
            batch_first=True)
        if self.bi:
            self.fc = nn.Linear(2*self.enc_units, nClasses)
        else:
            self.fc = nn.Linear(self.enc_units, nClasses)
        self.W1 = nn.Linear(self.enc_units, self.dec_units) #512x512
        self.W2 = nn.Linear(self.enc_units, self.dec_units) #512x512
        self.V = nn.Linear(self.enc_units, 1) #512X1

    def forward(self, x, hidden, enc_output):
        enc_output = enc_output.permute(1, 0, 2) #TBH
        hidden = torch.sum(hidden, 0)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden)) #
        attention_weights = torch.softmax(self.V(score), dim=1)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=0)
        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(0), x), -1)
        output, state = self.rnn(x)
        output =  output.view(-1, output.size(2))
        x = self.fc(output)
        return x, state, attention_weights

    # def init_hidden(self, batch_size):
    #     return torch.zeros((1, batch_size, self.dec_units))

class Seq2Seq(object):
    def __init__(self, nClasses, embed_dim, enc_units, dec_units):
        self.enc = EncoderRNN(nClasses, embed_dim, enc_units)
        self.dec = RNN_Decoder(nClasses, embed_dim, enc_units, dec_units)
        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()

        params = list(self.enc.parameters())
        params += list(self.dec.parameters())
        self.parameters = params

    def train(self):
        self.enc.train()
        self.dec.train()

    def eval(self):
        self.enc.eval()
        self.dec.eval()

    def state_dict(self):
        state_dict = [self.enc.state_dict(), self.dec.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.enc.load_state_dict(state_dict[0])
        self.dec.load_state_dict(state_dict[1])

    def enc(self, x, lengths):
        return self.enc(x, lengths)

    def dec(self, x, hidden, enc_out):
        return self.dec(x, hidden, enc_output)
