import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super().__init__()
class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super().__init__()
class GatedGCN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super().__init__()

def weights_init_uniform(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.uniform_(-0.01, 0.01)
            if m.bias is not None:
                m.bias.data.uniform_(-0.01, 0.01)

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.left_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
            for i in range(num_layers)])
        self.right_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
            for i in range(num_layers)])
        self.self_loof_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
            for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, x):
        nadj = adj[:, 0]
        neigh = (nadj == 0.5).float()
        rneigh = (nadj == 1.0).float()
        denom = neigh.sum(-1, keepdim=True) + 1 + rneigh.sum(-1, keepdim=True)
        for l in range(self.num_layers):
            self_node = self.self_loof_layers[l](x)
            left_neigh_Ax = self.left_gcn_layers[l](torch.einsum('ijkl, ijlz -> ijkz', neigh, x))
            right_neigh_Ax = self.right_gcn_layers[l](torch.einsum('ijkl, ijlz -> ijkz', rneigh, x))
            if l != self.num_layers - 1:
                AxW = (self_node + left_neigh_Ax + right_neigh_Ax) / denom
            else:
                AxW = self_node + left_neigh_Ax  + right_neigh_Ax
            gAxWb = torch.relu(AxW)
            x = self.dropout(gAxWb) if l < self.num_layers - 1 else gAxWb
        return x

class DynamicAttLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, 
            dropout, bi, num_layers, device):  
        super().__init__()
        self.layer = DynamicLSTM(input_size, hidden_size, 
            dropout, bi, num_layers, device)
        self.attention = Attention(hidden_size*2)

    def forward(self, x, x_len):
        out = self.layer(x, x_len)
        out, alpha = self.attention(out, out, x_len)
        return out, alpha

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, 
            dropout, bi, num_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            batch_first=True, bidirectional=bi, num_layers=num_layers, 
            dropout=dropout)
        self.bi = bi
        self.num_layers = num_layers

    def _init_hidden(self, size, hidden_size):
        hidden = (
            torch.zeros((2 if self.bi else 1)*self.num_layers, size, hidden_size, device=self.device), 
            torch.zeros((2 if self.bi else 1)*self.num_layers, size, hidden_size, device=self.device))
        return hidden

    def forward(self, x, x_len):
        hidden = self._init_hidden(len(x), self.hidden_size)
        x_pack, x_unsort_idx = self.sort_pack(x, x_len)
        self.layer.flatten_parameters()
        out, _ = self.layer(x_pack, hidden)
        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out[0][x_unsort_idx]
        return out

    def sort_pack(self, x, x_len):
        x_sort_idx = torch.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        x_unsort_idx = torch.argsort(x_sort_idx)
        x_pack = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        return x_pack, x_unsort_idx

class Attention(nn.Module):
    def __init__(self, hid_size):
        super(Attention, self).__init__()
        self.att = nn.Linear(hid_size, 1, bias=False)

    def forward(self, inp, ainp, len_s):
        logit = self.att(ainp).squeeze(-1)
        weight = self._masked_softmax(logit, len_s)
        weighted_out = weight.unsqueeze(1).bmm(inp).squeeze(1)
        return weighted_out, weight

    def _masked_softmax(self, mat, len_s):
        len_s = len_s.long()
        idxes = torch.arange(0, int(mat.size(-1)), out=mat.data.new(int(len_s.max())).long()).unsqueeze(1)
        mask = (idxes < len_s.unsqueeze(0)).float().permute(1, 0).requires_grad_(False)
        zero_vec = -9e15 * torch.ones_like(mask)
        attention = torch.where(mask > 0, mat, zero_vec)
        attention = torch.softmax(attention, dim=-1)
        return attention