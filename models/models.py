import torch
import torch.nn as nn
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, layers, d_out, dropout, activation):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(dropout)

        for ind, (l_in, l_out) in enumerate(layers):
            self.mlp.add_module('BN_{}'.format(ind), nn.BatchNorm1d(l_in))
            self.mlp.add_module('Linear_{}'.format(ind), nn.Linear(l_in, l_out))
            self.mlp.add_module('Activation_{}'.format(ind), activation)
            self.mlp.add_module('Dropout_{}'.format(ind), self.dropout)
        self.mlp.add_module('BN_Out', nn.BatchNorm1d(layers[-1][1]))
        self.mlp.add_module('Linear_Out', nn.Linear(layers[-1][1], d_out))

    def forward(self, X):
        return self.mlp(X)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.d_embed,
                           hidden_size=config.d_hidden,
                           num_layers=config.n_layers,
                           batch_first=False,
                           dropout=config.dropout_rnn,
                           bidirectional=config.bidir)

    def forward(self, X):
        batch_size = X.size(1)
        state_shape = (self.config.n_cells, batch_size, self.config.d_hidden)
        h0 = c0 = Variable(X.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(X, (h0, c0))
        # print('Hidden Size: ', ht.size())
        if not self.config.bidir:
            return ht[-1]
        else:
            return ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class ConcatModel(nn.Module):
    def __init__(self, config):
        super(ConcatModel, self).__init__()

        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.encoder = Encoder(config)
        self.relu = nn.ReLU()

        seq_in_size = 2 * config.d_hidden
        if config.bidir:
            seq_in_size *= 2
        layers = [[seq_in_size] * 2] * config.n_linear_layers

        self.out = MLP(layers, config.d_out, config.dropout_mlp, self.relu)

    def forward(self, X):
        premise = self.embed(X.premise)
        hypothesis = self.embed(X.hypothesis)

        premise = self.encoder(premise)
        hypothesis = self.encoder(hypothesis)

        return self.out(torch.cat([premise, hypothesis], 1))
