import torch
from torch._C import device
import torch.nn as nn
import numpy as np


class TransformationModel(nn.Module):
    def __init__(self, D_in=21 * 3, H=256, D_out=15 * 3, device="cpu"):
        """
        Crate a two-layers networks with relu activation.
        """
        super(TransformationModel, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(D_in, H)  # , bias=False)
        self.linear2 = torch.nn.Linear(H, H)  # , bias=False)
        self.linear3 = torch.nn.Linear(H, H)  # , bias=False)

        self.rot_head = torch.nn.Linear(H, D_out)
        self.tran_head = torch.nn.Linear(H, D_out)

        self.actvn = nn.LeakyReLU(0.1)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        zero = torch.zeros([x.shape[0], 1, 3], device=x.device)

        x = x.reshape(-1, 21 * 3)
        x = self.linear1(x)
        x = self.linear2(self.actvn(x)) + x
        x = self.linear3(self.actvn(x)) + x

        # y_pred = x.reshape(-1, 21, 3)
        # import pdb; pdb.set_trace()
        rot_out = self.rot_head(x)
        rot_out = rot_out.reshape(-1, 15, 3)
        rot_out = torch.cat([zero, rot_out], 1)

        tran_out = self.tran_head(x)
        tran_out = tran_out.reshape(-1, 15, 3)
        tran_out = torch.cat([zero, tran_out], 1)
        return rot_out, tran_out


def get_transformation_layer(model_path):
    model = torch.load(model_path)
    return model
