import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimpleEncoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        Crate a two-layers networks with relu activation.
        """
        super(SimpleEncoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.out_layer = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h_relu).clamp(min=0)
        y_pred = self.out_layer(h2_relu)
        return y_pred


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recode to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class LetentEncoder(nn.Module):
    ''' Latent encoder class.
    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.
    Args:
        z_dim (int): dimension of output code z
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension, joint_num
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=64, c_dim=128, in_dim=21, dim=128, leaky=True):  # leaky=True
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fc_pos = nn.Linear(in_dim, dim)

        # if c_dim != 0:
        #     self.fc_c = nn.Linear(c_dim, 128)

        # self.fc_0 = nn.Linear(1, 128)
        self.fc_0 = nn.Linear(in_dim * 3 + c_dim, dim)
        self.fc_1 = nn.Linear(dim, dim)
        self.fc_2 = nn.Linear(dim, dim)
        self.fc_3 = nn.Linear(dim, dim)
        self.fc_mean = nn.Linear(dim, z_dim)
        self.fc_logstd = nn.Linear(dim, z_dim)

        if not leaky:
            self.actvn = F.relu
            self.pool = maxpool
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            self.pool = torch.mean

    def forward(self, joints, c, **kwargs):
        # batch_size, 21, 3
        batch_size, joint_len, D = joints.size()

        # output size: B x T X F
        # net = self.fc_0(x.unsqueeze(-1))
        # net = net + self.fc_pos(p)

        joints = joints.reshape(batch_size, -1)
        net = self.fc_0(torch.cat([joints, c], dim=1))

        net = self.fc_1(self.actvn(net)) + net
        net = self.fc_2(self.actvn(net)) + net
        net = self.fc_3(self.actvn(net)) + net

        # if self.c_dim != 0:
        #     net = net + self.fc_c(c).unsqueeze(1)

        # net = self.fc_1(self.actvn(net))
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)

        # net = self.fc_2(self.actvn(net))
        # pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # net = torch.cat([net, pooled], dim=2)

        # net = self.fc_3(self.actvn(net))
        # # Reduce
        # #  to  B x F
        # net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd
