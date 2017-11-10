import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.d_embed,
                           hidden_size=config.d_hidden,
                           num_layers=config.n_layers,
                           batch_first=False,
                           dropout=config.dropout,
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
        # self.encoder = nn.LSTM(config.d_embed,
        #                        config.d_hidden,
        #                        config.n_layers,
        #                        bias=True,
        #                        batch_first=True,
        #                        bidirectional=config.bidir,
        #                        dropout=config.dropout)
        self.encoder = Encoder(config)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.dropout)

        seq_in_size = 2 * config.d_hidden
        if config.bidir:
            seq_in_size *= 2
        lin_config = [seq_in_size] * 2
        self.out = nn.Sequential(nn.Linear(*lin_config),
                                 self.relu,
                                 self.dropout,
                                 nn.Linear(*lin_config),
                                 self.relu,
                                 self.dropout,
                                 nn.Linear(*lin_config),
                                 self.relu,
                                 self.dropout,
                                 nn.Linear(seq_in_size, config.d_out))

    def forward(self, X):
        # print('Input_size: ', X.premise.size())
        premise = self.embed(X.premise)
        # print('Premise Emb Size: ', premise.size())
        hypothesis = self.embed(X.hypothesis)
        # print('Hypothesis Emb Size: ', hypothesis.size())
        premise = self.encoder(premise)
        # print('Premise Encoder Size: ', premise.size())
        hypothesis = self.encoder(hypothesis)

        return self.out(torch.cat([premise, hypothesis], 1))
