import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cosine


def get_masks(input, pad_token=1):
    return (input != pad_token).unsqueeze(2).float()

def softmask(input, mask):
    """
    input is of dims batch_size x len_1 x len_2
    mask if of dims batch_size x len_2 x 1
    """
    batch_size, len_1, len_2 = input.size()

    assert len_2 == mask.size(1)

    exp_input = torch.exp(input)
    divisors = torch.bmm(exp_input, mask).view(batch_size, len_1)  # batch_size x len_1
    masked = torch.mul(exp_input, torch.transpose(mask.expand(batch_size, len_2, len_1), 1, 2))

    return masked.div(divisors.unsqueeze(2).expand_as(masked))  # batch_size x len_1 x len_2


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

class DA_MLP(nn.Module):
    def __init__(self, layers, dropout, activation):
        super(DA_MLP, self).__init__()
        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(dropout)

        for ind, (l_in, l_out) in enumerate(layers):
            self.mlp.add_module('Dropout_{}'.format(ind), self.dropout)
            self.mlp.add_module('Linear_{}'.format(ind), nn.Linear(l_in, l_out))
            self.mlp.add_module('Activation_{}'.format(ind), activation)

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

# class ConcatModel(nn.Module):
#     def __init__(self, config):
#         super(ConcatModel, self).__init__()

#         self.config = config
#         self.embed = nn.Embedding(config.n_embed, config.d_embed)
#         self.encoder = Encoder(config)
#         self.relu = nn.ReLU()

#         seq_in_size = 2 * config.d_hidden
#         if config.bidir:
#             seq_in_size *= 2
#         layers = [[seq_in_size] * 2] * config.n_linear_layers

#         self.out = MLP(layers, config.d_out, config.dropout_mlp, self.relu)

#     def forward(self, X):
#         premise = self.embed(X.premise)
#         hypothesis = self.embed(X.hypothesis)

#         premise = self.encoder(premise)
#         hypothesis = self.encoder(hypothesis)

#         return self.out(torch.cat([premise, hypothesis], 1))


class ConcatModel(nn.Module):
    def __init__(self, config):
        super(ConcatModel, self).__init__()

        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.cell = EncoderCell(config.d_embed, config.d_hidden)
        self.relu = nn.ReLU()

        seq_in_size = 2 * config.d_hidden
        if config.bidir:
            seq_in_size *= 2
        layers = [[seq_in_size] * 2] * config.n_linear_layers

        self.out = MLP(layers, config.d_out, config.dropout_mlp, self.relu)

    def forward(self, X):
        premise = self.embed(X.premise)
        hypothesis = self.embed(X.hypothesis)

        batch_size = premise.size(1)

        # Combine Premise and Hypothesis
        combined = torch.cat([premise, hypothesis], dim=1)

        # Forward Pass
        h_fw, c_fw = self.hidden_init(batch_size, self.config.d_embed)
        premise_fw, hypothesis_fw = [], []
        for word_input in torch.cat([premise, hypothesis], dim=1):
            h_fw, c_fw = self.cell(word_input, h_fw, c_fw)
            premise_fw.append(h_fw[:batch_size])
            hypothesis_fw.append(h_fw[batch_size:])

        # Backward Pass
        h_bw, c_bw = self.hidden_init(batch_size, self.config.d_embed)
        premise_bw, hypothesis_bw = [], []
        for ind in range(combined.size(0) - 1, -1, -1):
            h_bw, c_bw = self.cell(word_input[ind], h_bw, c_bw)
            premise_bw.append(h_bw[:batch_size])
            hypothesis_bw.append(h_bw[batch_size:])

        print(len(premise_bw), len(hypothesis_bw), len(premise_fw), len(hypothesis_fw))

        # return self.out(torch.cat([premise, hypothesis], dim=1))
        return self.out(torch.cat([premise_fw[-1],
                                   premise_bw[-1],
                                   hypothesis_fw[-1],
                                   hypothesis_bw[-1]], dim=1))

    def hidden_init(self, batch_size, d_embed):
        c, h = (Variable(torch.zeros(batch_size * 2, self.config.d_hidden)),
                Variable(torch.zeros(batch_size * 2, self.config.d_hidden)))
        if self.config.cuda:
            c, h = c.cuda(), h.cuda()
        return (c, h)


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
        self.emb_drop = nn.Dropout(p=config.dropout_emb)
        self.dropout = nn.Dropout(p=config.dropout_mlp)
        self.mlp = nn.Linear(self.dim*8, self.dim, bias=True)
        self.cl = nn.Linear(self.dim, 3)
        self.premise = nn.LSTM(input_size = self.embedding_dim,
                               hidden_size = self.dim,
                               num_layers = config.n_layers,
                               dropout = config.dropout_rnn,
                               bidirectional=True)
        self.hypothesis = nn.LSTM(input_size = self.embedding_dim,
                                  hidden_size = self.dim,
                                  num_layers = config.n_layers,
                                  dropout = config.dropout_rnn,
                                  bidirectional=True)
        self.v1 = nn.LSTM(input_size = self.dim*8,
                          hidden_size = self.dim,
                          num_layers = config.n_layers,
                          dropout = config.dropout_rnn,
                          bidirectional=True)
        self.v2 = nn.LSTM(input_size = self.dim*8,
                          hidden_size = self.dim,
                          num_layers = config.n_layers,
                          dropout = config.dropout_rnn,
                          bidirectional=True)

    def forward(self, x):

        p = x.premise #[sentence_length x batch_size]
        h = x.hypothesis #[sentence_length x batch_size]

        ### Get max sequence lengths ###
        p_length = p.size(0)
        h_length = h.size(0)
    
        ### Get masks for true sequence length ###
        mask_p = get_masks(p)
        mask_h = get_masks(h)
        
        ### Embed inputs ###
        p = self.emb_drop(self.embed(p))
        h = self.emb_drop(self.embed(h))

        ### First biLSTM layer ###
        premise_bi, states = self.premise(p)
        hypothesis_bi, states = self.hypothesis(h)

        # def unstack(tensor, dim):
        #     return [torch.squeeze(x) for x in torch.split(tensor, 1, dim=dim)]
        #
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
            alpha_i = torch.exp(scores_i).mul(mask_h)         
            alpha_i = alpha_i / alpha_i.sum(dim=0).expand_as(alpha_i)
 
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
            beta_j = torch.exp(scores_j).mul(mask_p)
            beta_j = beta_j / beta_j.sum(dim=0).expand_as(beta_j)
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

class DA(nn.Module):
    # TODO  Out-of-vocabulary (OOV) words are
    # hashed to one of 100 random embeddings
    # each initialized to mean 0 and standard
    # deviation 1.
    def __init__(self, config):
        super(DA, self).__init__()

        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)

        self.intra_sentence = config.intra_sentence

        # So embedding doesn't change
        self.embed.weight.requires_grad = False

        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()

        self.d_embed = config.d_embed
        self.d_hidden = config.d_hidden

        self.in_linear = nn.Linear(self.d_embed, self.d_hidden, bias=False)

        def _mlp_layers(input_dim, output_dim):
            mlp_layers = []
            mlp_layers.append(nn.Linear(
                input_dim, output_dim, bias=True))
            mlp_layers.append(self.relu)
            mlp_layers.append(nn.BatchNorm1d(output_dim))
            mlp_layers.append(nn.Dropout(p=0.2))
            mlp_layers.append(nn.Linear(
                output_dim, output_dim, bias=True))
            mlp_layers.append(self.relu)
            mlp_layers.append(nn.BatchNorm1d(output_dim))
            mlp_layers.append(nn.Dropout(p=0.2))
            return nn.Sequential(*mlp_layers)  # * used to unpack list

        self.mlp_F = _mlp_layers(self.d_hidden, self.d_hidden)
        self.mlp_G = _mlp_layers(self.d_hidden * 2, self.d_hidden)
        self.mlp_H = _mlp_layers(self.d_hidden * 2, self.d_hidden)

        if self.intra_sentence:
            self.mlp_intra = DA_MLP(((self.d_hidden, self.d_hidden), (self.d_hidden, self.d_hidden)), config.dropout_mlp, self.relu)
            self.mlp_F = DA_MLP(((self.d_hidden * 2, self.d_hidden), (self.d_hidden, self.d_hidden)), config.dropout_mlp, self.relu)
        else:
            self.mlp_F = DA_MLP(((self.d_hidden, self.d_hidden),(self.d_hidden, self.d_hidden)), config.dropout_mlp, self.relu)

        self.mlp_G = DA_MLP(((self.d_hidden * 2, self.d_hidden),(self.d_hidden, self.d_hidden)), config.dropout_mlp, self.relu)
        self.mlp_H = DA_MLP(((self.d_hidden * 2, self.d_hidden),(self.d_hidden, self.d_hidden)), config.dropout_mlp, self.relu)

        self.out_linear = nn.Linear(self.d_hidden, config.d_out)

        '''initialize parameters'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)

    def forward(self, X):

        p = torch.transpose(X.premise, 0, 1)
        h = torch.transpose(X.hypothesis, 0, 1)

        batch_size = p.size(0)

        ### Get max sequence lengths ###
        p_length = p.size(1)
        h_length = h.size(1)

        ### Get masks for true sequence length ###
        p_mask = get_masks(p)
        h_mask = get_masks(h)

        # Embed premise & hypothesis
        p_embedded = self.embed(p)
        h_embedded = self.embed(h)

        # Normalize embeddings
        # norm = p_embedded.norm(p=2, dim=2, keepdim=True)
        # p_embedded = p_embedded.div(norm.expand_as(p_embedded))
        #
        # norm = h_embedded.norm(p=2, dim=2, keepdim=True)
        # h_embedded = h_embedded.div(norm.expand_as(h_embedded))

        # Apply initial linear encoding
        p_linear = self.in_linear(p_embedded.view(batch_size * p_length, self.d_embed)).view(batch_size, p_length, self.d_hidden)
        h_linear = self.in_linear(h_embedded.view(batch_size * h_length, self.d_embed)).view(batch_size, h_length, self.d_hidden)

        if self.intra_sentence:
            p_intra = self.mlp_intra(p_linear.view(batch_size * p_length, self.d_hidden)).view(batch_size, p_length, self.d_hidden)
            h_intra = self.mlp_intra(h_linear.view(batch_size * h_length, self.d_hidden)).view(batch_size, h_length, self.d_hidden)

            # Self Attend
            p_self_scores = torch.bmm(p_intra, torch.transpose(p_intra, 1, 2))
            p_probs = softmask(p_self_scores, p_mask)
            p_intra = torch.cat((p_linear, torch.bmm(p_probs, p_linear)), 2)

            h_self_scores = torch.bmm(h_intra, torch.transpose(h_intra, 1, 2))
            h_probs = softmask(h_self_scores, h_mask)
            h_intra = torch.cat((h_linear, torch.bmm(h_probs, h_linear)), 2)

            # Apply F
            F_p = self.mlp_F(p_intra.view(batch_size * p_length, -1)).view(batch_size, p_length, self.d_hidden)
            F_h = self.mlp_F(h_intra.view(batch_size * h_length, -1)).view(batch_size, h_length, self.d_hidden)

        else:
            # Apply F
            F_p = self.mlp_F(p_linear.view(batch_size * p_length, -1)).view(batch_size, p_length, self.d_hidden)
            F_h = self.mlp_F(h_linear.view(batch_size * h_length, -1)).view(batch_size, h_length, self.d_hidden)

        # Attend
        sim_scores = torch.bmm(F_p, torch.transpose(F_h, 1, 2))

        p_probs = softmask(sim_scores, h_mask)
        h_probs = softmask(torch.transpose(sim_scores, 1, 2), p_mask)

        h_attended = torch.bmm(p_probs, h_linear)
        p_attended = torch.bmm(h_probs, p_linear)

        ###Combine
        combined_p = torch.cat((p_linear, h_attended), -1)
        combined_h = torch.cat((h_linear, p_attended), -1)

        #Apply G
        G_p = self.mlp_G(combined_p.view(batch_size * p_length, 2 * self.d_hidden))
        G_h = self.mlp_G(combined_h.view(batch_size * h_length, 2 * self.d_hidden))

        p_output = torch.sum(G_p.view(batch_size, p_length, self.d_hidden), 1)
        h_output = torch.sum(G_h.view(batch_size, h_length, self.d_hidden), 1)

        # Apply H
        output = self.mlp_H(torch.cat((p_output, h_output), -1))
        output = self.out_linear(output)

        return output
