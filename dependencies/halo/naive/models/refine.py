import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RefineNet(nn.Module):
    ''' RefineNet class.
    Takes noisy joints and object latent vector as input and output the refined joints
    Args:
        out_dim (int): dimension of output code z
        c_dim (int): dimension of object latent code c
        dim (int): input dimension, joint_num
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, in_dim=21, c_dim=128, out_dim=21 * 3, dim=128):  # c_dim=128
        super().__init__()
        self.out_dim = out_dim
        self.c_dim = c_dim
        self.dist_dim = 21  #  5  # finger tip to surface distances

        # self.fc_0 = nn.Linear(1, 128)
        # Size: joints + object + joint-to-surface dist
        self.fc_0 = nn.Linear(in_dim * 3 + c_dim + self.dist_dim, dim)
        self.fc_1 = nn.Linear(dim, dim)
        self.fc_2 = nn.Linear(dim, dim)
        self.fc_3 = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, out_dim)

        self.actvn = nn.LeakyReLU(0.1)

    def forward(self, joints, obj_code, dist, **kwargs):
        # batch_size, 21, 3
        batch_size, joint_len, D = joints.size()

        # output size: B x T X F
        # net = self.fc_0(x.unsqueeze(-1))
        # net = net + self.fc_pos(p)

        joints = joints.reshape(batch_size, -1)
        net = self.fc_0(torch.cat([joints, obj_code, dist], dim=1))

        net = self.fc_1(self.actvn(net)) + net
        net = self.fc_2(self.actvn(net)) + net
        net = self.fc_3(self.actvn(net)) + net
        net = self.fc_out(net)

        y_pred = net.reshape(-1, 21, 3)

        return y_pred
