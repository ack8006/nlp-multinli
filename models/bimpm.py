import torch
import torch.nn as nn
from torch.autograd import Variable
from multi_perspective import MultiPerspective


class MLP(nn.Module):
    def __init__(self, layers, d_out, dropout, activation, batch_norm=True):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(dropout)

        for ind, (l_in, l_out) in enumerate(layers):
            if batch_norm:
                self.mlp.add_module('BN_{}'.format(ind), nn.BatchNorm1d(l_in))
            self.mlp.add_module('Linear_{}'.format(ind), nn.Linear(l_in, l_out))
            self.mlp.add_module('Activation_{}'.format(ind), activation)
            self.mlp.add_module('Dropout_{}'.format(ind), self.dropout)
        if batch_norm:
            self.mlp.add_module('BN_Out', nn.BatchNorm1d(layers[-1][1]))
        self.mlp.add_module('Linear_Out', nn.Linear(layers[-1][1], d_out))

    def forward(self, X):
        return self.mlp(X)


class Encoder(nn.Module):
    def __init__(self, d_input, d_hidden, n_layers, dropout):
        super(Encoder, self).__init__()
        self.d_hidden = d_hidden
        self.rnn = nn.LSTM(input_size=d_input,
                           hidden_size=d_hidden,
                           num_layers=n_layers,
                           batch_first=False,
                           dropout=dropout,
                           bidirectional=True)

    def forward(self, X):
        batch_size = X.size(1)
        state_shape = (2, batch_size, self.d_hidden)
        h0 = c0 = Variable(X.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(X, (h0, c0))
        # print('Hidden Size: ', ht.size())
        return ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class EncoderCell(nn.Module):
    '''
        @return: h1, c1
    '''

    def __init__(self, d_embed, d_hidden):
        super(EncoderCell, self).__init__()
        self.cell = nn.LSTMCell(input_size=d_embed,
                                hidden_size=d_hidden,
                                bias=True)

    def forward(self, X, h0, c0):
        return self.cell(X, (h0, c0))


class BiMPM(nn.Module):
    def __init__(self, config):
        super(BiMPM, self).__init__()

        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.cell = EncoderCell(config.d_embed, config.d_hidden)

        self.multi_perspective = MultiPerspective(config)
        self.aggregation_layer = Encoder(2 * 2 * 4 * config.mp_dim,
                                         config.agg_d_hidden,
                                         config.agg_n_layers,
                                         config.dropout_rnn)
        self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()

        # Bidirectional * n_layers * agg_hidden
        seq_in_size = 2 * config.agg_n_layers * config.agg_d_hidden
        layers = [[seq_in_size] * 2] * config.n_linear_layers

        self.out = MLP(layers, config.d_out, config.dropout_mlp, self.activation)

    def forward(self, X):
        premise = self.embed(X.premise)
        hypothesis = self.embed(X.hypothesis)

        batch_size = premise.size(1)

        # Combine Premise and Hypothesis
        combined = torch.cat([premise, hypothesis], dim=1)

        # Forward Pass
        h_fw, c_fw = self.hidden_init(batch_size)
        premise_fw, hypothesis_fw = [], []
        for word_input in combined:
            h_fw, c_fw = self.cell(word_input, h_fw, c_fw)
            premise_fw.append(h_fw[:batch_size])
            hypothesis_fw.append(h_fw[batch_size:])

        # Backward Pass
        h_bw, c_bw = self.hidden_init(batch_size)
        premise_bw, hypothesis_bw = [], []
        for ind in range(combined.size(0) - 1, -1, -1):
            h_bw, c_bw = self.cell(combined[ind], h_bw, c_bw)
            premise_bw.append(h_bw[:batch_size])
            hypothesis_bw.append(h_bw[batch_size:])

        premise_fw = torch.stack(premise_fw)
        premise_bw = torch.stack(premise_bw)
        hypothesis_fw = torch.stack(hypothesis_fw)
        hypothesis_bw = torch.stack(hypothesis_bw)
        # print('prem_size', premise_fw.size())

        matchings_ph = self.multi_perspective(premise_fw, premise_bw, hypothesis_fw, hypothesis_bw)
        matchings_hp = self.multi_perspective(hypothesis_fw, hypothesis_bw, premise_fw, premise_bw)
        matchings = torch.cat([matchings_ph, matchings_hp], dim=-1)  # (sentence_len, batch_size, 2 * 2 * 4 * mp_dim)
        # print('Matching Size: ', matchings.size())

        aggregation = self.aggregation_layer(matchings)
        # print('Aggregation Size: ', aggregation.size())

        return self.out(aggregation)

    def hidden_init(self, batch_size):
        if self.config.cuda:
            return (Variable(torch.zeros(batch_size * 2, self.config.d_hidden).cuda()),
                    Variable(torch.zeros(batch_size * 2, self.config.d_hidden).cuda()))
        else:
            return (Variable(torch.zeros(batch_size * 2, self.config.d_hidden)),
                    Variable(torch.zeros(batch_size * 2, self.config.d_hidden)))
