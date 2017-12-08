import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CustomWeight(nn.Module):
    def __init__(self, d_hidden, mp_dim, cuda):
        super(CustomWeight, self).__init__()
        self.d_hidden = d_hidden
        self.mp_dim = mp_dim
        self.weight = Variable(torch.Tensor(1, mp_dim, d_hidden), requires_grad=True)
        self.weight_init()
        if cuda:
            self.weight = self.weight.cuda()

    def forward(self, X):
        # print('Custom Weight')
        timesteps = X.size(0)
        X = X.view(-1, X.size(-1))
        # print(X.size())
        X = torch.unsqueeze(X, dim=1)
        # print('X and W Size: ', X.size(), self.weight.size())
        X = X * self.weight
        # print('Out Size: ', X.view(timesteps, -1, self.mp_dim, self.d_hidden).size(), '\n')
        return X.view(timesteps, -1, self.mp_dim, self.d_hidden)

    def weight_init(self):
        nn.init.xavier_uniform(self.weight)


class MultiPerspective(nn.Module):
    # def __init__(self, input_shape, mp_dim, epsilon=1e-6):
    def __init__(self, config):
        super(MultiPerspective, self).__init__()
        self.config = config
        self.weight_fm_fw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)
        self.weight_fm_bw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)
        self.weight_mpm_fw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)
        self.weight_mpm_bw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)
        self.weight_am_fw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)
        self.weight_am_bw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)
        self.weight_mam_fw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)
        self.weight_mam_bw = CustomWeight(self.config.d_hidden, self.config.mp_dim, self.config.cuda)

    def forward(self, p_fw, p_bw, h_fw, h_bw):
        ''' All Attentive Vectors are size (sentence_length, batch_size) '''
        # 4 matching strategy
        # list_matching = []

        fm_fw = self._full_matching(p_fw, h_fw, self.weight_fm_fw)
        fm_bw = self._full_matching(p_bw, h_bw, self.weight_fm_bw)

        mpm_fw = self._maxpooling_matching(p_fw, h_fw, self.weight_mpm_fw)
        mpm_bw = self._maxpooling_matching(p_bw, h_bw, self.weight_mpm_bw)

        # cos_matrix, (premise_time, hypothesis_time, batch)
        cos_matrix_fw = self.cosine_similarity(torch.unsqueeze(p_fw, dim=1),
                                               torch.unsqueeze(h_fw, dim=0))
        cos_matrix_bw = self.cosine_similarity(torch.unsqueeze(p_bw, dim=1),
                                               torch.unsqueeze(h_bw, dim=0))

        ma_fw = self._attentive_matching(p_fw, h_fw, self.weight_am_fw, cos_matrix_fw)
        ma_bw = self._attentive_matching(p_bw, h_bw, self.weight_am_bw, cos_matrix_bw)

        mam_fw = self._max_attentive_matching(p_fw, h_fw, self.weight_mam_fw, cos_matrix_fw)
        mam_bw = self._max_attentive_matching(p_bw, h_bw, self.weight_mam_bw, cos_matrix_bw)

        return torch.cat([fm_fw, fm_bw, mpm_fw, mpm_bw, ma_fw, ma_bw, mam_fw, mam_bw], dim=-1)

    def cosine_similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2, dim=-1)

    def _full_matching(self, x1, x2, fm_layer):
        ''' (1) Full-Matching
        # Arguments
            h1: (time_steps, batch_size, d_hidden)
            h2: (time_steps, batch_size, d_hidden)
        # Output shape
            (time_steps, batch_size, mp_dim)
        '''
        # print('FM Sizes')
        # print(x1.size(), x2.size(), x2[-1:].size())
        x1 = fm_layer(x1)
        x2 = fm_layer(x2[-1:])  # x2[-1:] -> (1, batch_size, d_hidden)
        # print(x1.size(), x2.size())
        return self.cosine_similarity(x1, x2)  # (time_steps, batch_size)

    def _maxpooling_matching(self, x1, x2, mpm_layer):
        x1 = torch.unsqueeze(mpm_layer(x1), dim=1)
        x2 = torch.unsqueeze(mpm_layer(x2), dim=0)
        return torch.max(self.cosine_similarity(x1, x2), dim=1)[0]

    def _attentive_matching(self, x1, x2, am_layer, cos_matrix):
        cos_matrix_exp = torch.unsqueeze(cos_matrix, dim=-1)  # (premise_time, hypothesis_time, batch, 1)
        x2 = torch.unsqueeze(x2, dim=0)  # (1, hypothesis_time, batch, d_hidden)
        mean_attentive_vector = torch.sum(cos_matrix_exp * x2, dim=1)  # (premise_time, batch, d_hidden)

        # PAPER DOESN'T SAY ABS, BUT MAY MAKE MORE SENSE AS WEIGHTED SUM
        cos_matrix_sum = torch.sum(cos_matrix, dim=1)  # (premise_time, batch)
        # cos_matrix_sum = torch.sum(torch.abs(cos_matrix), dim=1)  # (premise_time, batch)
        cos_matrix_sum = torch.unsqueeze(cos_matrix_sum, dim=-1)  # (premise_time, batch, 1)

        mean_attentive = mean_attentive_vector / cos_matrix_sum  # (premise_time, batch, d_hidden)
        x1 = am_layer(x1)
        mean_attentive = am_layer(mean_attentive)
        return self.cosine_similarity(x1, mean_attentive)

    def _max_attentive_matching(self, x1, x2, mam_layer, cos_matrix):
        index_mask = torch.max(cos_matrix, dim=1)[1]
        timesteps = x1.size(0)
        batch_size = x1.size(1)
        # All of this basically gets the index of the flattened x2 matrix from the max
        index_mask = index_mask * batch_size
        index_mask += Variable(torch.Tensor(list(range(batch_size))).long())
        index_mask = index_mask.view(-1)
        max_attentive = torch.index_select(x2.view(-1, self.config.d_hidden), 0, index_mask)
        max_attentive = max_attentive.view(timesteps, batch_size, -1)
        x1 = mam_layer(x1)
        max_attentive = mam_layer(max_attentive)
        return self.cosine_similarity(x1, max_attentive)



















