import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.spatial.distance import cosine


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


class CosineModel(nn.Module):
    def __init__(self, config):
        super(CosineModel, self).__init__()

        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.encoder = Encoder(config)
        self.relu = nn.ReLU()

        seq_in_size = 2 * config.d_hidden
        if config.bidir:
            seq_in_size *= 2
        seq_in_size += 1
        layers = [[seq_in_size] * 2] * config.n_linear_layers

        self.out = MLP(layers, config.d_out, config.dropout_mlp, self.relu)

    def forward(self, X):
        premise = self.embed(X.premise)
        hypothesis = self.embed(X.hypothesis)

        premise = self.encoder(premise)
        hypothesis = self.encoder(hypothesis)
        dist = self.calculate_distances(premise, hypothesis)

        return self.out(torch.cat([dist, premise, hypothesis], 1))

    def calculate_distances(self, x1, x2):
        distances = torch.Tensor(x1.size(0), 1).float()
        for d in range(distances.size(0)):
            distances[d, 0] = cosine(x1[d].data.cpu().numpy(), x2[d].data.cpu().numpy())
        if self.config.cuda:
            distances = distances.cuda()
        return Variable(distances)

class ESIM(nn.Module):
    def __init__(self, config):
        super(ESIM, self).__init__()
        self.config = config
        self.embedding_dim = config.d_embed
        self.dim = config.d_hidden
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.emb_drop = nn.Dropout(p=config.dropout_rnn)
        self.dropout = nn.Dropout(p=config.dropout_mlp)
        self.mlp = nn.Linear(self.dim*8, self.dim, bias=True)
        self.cl = nn.Linear(self.dim, 3)
        self.premise = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.dim, bidirectional=True)
        self.hypothesis = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.dim, bidirectional=True)
        self.v1 = nn.LSTM(input_size = self.embedding_dim*8, hidden_size = self.dim, bidirectional=True)
        self.v2 = nn.LSTM(input_size = self.embedding_dim*8, hidden_size = self.dim, bidirectional=True)

    def forward(self, p, h):

        ### Get max sequence lengths ###
        p_length = p.size(0)
        h_length = h.size(0)
    
        ### Get masks for true sequence length ###
        mask_p = p != 1
        mask_h = h != 1
        
        ### Embed inputs ###
        p = self.emb_drop(self.embed(p))
        h = self.emb_drop(self.embed(h))

        ### First biLSTM layer ###
        premise_bi, states = self.premise(p)
        hypothesis_bi, states = self.hypothesis(h)

        def unstack(tensor, dim):
            return [torch.squeeze(x) for x in torch.split(tensor, 1, dim=dim)]
            
        premise_list = unstack(premise_bi, 0)
        hypothesis_list = unstack(hypothesis_bi, 0)
                
        ### Attention ###
        scores_all = []
        premise_attn = []
        alphas = []
        
        for i in range(p_length):

            scores_i_list = []
            for j in range(h_length):
                score_ij = torch.sum(premise_list[i].mul(hypothesis_list[j]), dim=1, keepdim=True)
                scores_i_list.append(score_ij)
                                        
            scores_i = torch.transpose(torch.stack(scores_i_list, dim=1), 0, 1)
            alpha_i = F.softmax(scores_i) # Masked?
    
            a_tilde_i = torch.sum(torch.mul(alpha_i, hypothesis_bi), 0)
    
            premise_attn.append(a_tilde_i)
            
            scores_all.append(scores_i)
            alphas.append(alpha_i)
        
        scores_stack = torch.stack(scores_all, dim=2)
        scores_list = unstack(scores_stack, dim=0)

        hypothesis_attn = []
        betas = []
        for j in range(h_length):
            scores_j = torch.transpose(scores_list[j], 0, 1).unsqueeze(2)
            beta_j = F.softmax(scores_j) # Masked?
            b_tilde_j = torch.sum(torch.mul(beta_j, premise_bi), 0)
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)
            
        ### Make attention-weighted sentence representations into one tensor ###
        premise_attns = torch.stack(premise_attn, dim=0)
        hypothesis_attns = torch.stack(hypothesis_attn, dim=0)

        ### Subcomponent Inference ###
        prem_diff = premise_bi.sub(premise_attns)
        prem_mul = premise_bi.mul(premise_attns)
        hyp_diff = hypothesis_bi.sub(hypothesis_attns)
        hyp_mul = hypothesis_bi.mul(hypothesis_attns)

        m_a = torch.cat((premise_bi, premise_attns, prem_diff, prem_mul), dim=2)
        m_b = torch.cat((hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul), dim=2)
        
        ### Inference Composition ###
        v1_outs, c3 = self.v1(m_a)
        v2_outs, c4 = self.v2(m_b)

        ### Pooling ###
        v_1_ave = torch.mean(v1_outs, 0)
        v_2_ave = torch.mean(v2_outs, 0)
        v_1_max = torch.max(v1_outs, 0)[0]
        v_2_max = torch.max(v2_outs, 0)[0]
        
        v = torch.cat((v_1_ave, v_2_ave, v_1_max, v_2_max), dim=1)

        ### MLP ###
        h_mlp = F.tanh(self.mlp(v))
        h_mlp = self.dropout(h_mlp)
        output = self.cl(h_mlp)

        return output
