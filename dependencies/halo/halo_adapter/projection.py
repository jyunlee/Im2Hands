import torch
import torch.nn as nn
import numpy as np


class JointProjectionLayer(nn.Module):
    def __init__(self, D_in=21 * 3, H=256, D_out=21 * 3, device="cpu", fix_root=True):
        """
        Crate a two-layers networks with relu activation.
        """
        super(JointProjectionLayer, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(D_in, H)  # , bias=False)
        self.linear2 = torch.nn.Linear(H, H)  # , bias=False)
        self.linear3 = torch.nn.Linear(H, D_out)  # , bias=False)

        self.actvn = nn.LeakyReLU(0.1)

        self.fix_root = fix_root
        self.endpoints = np.array([0])
        self.endpoints_flat = []
        for idx in self.endpoints:
            for j in range(3):
                self.endpoints_flat.append(idx * 3 + j)
        self.endpoints_flat = np.array(self.endpoints_flat)
        # print("end point flat = ", self.endpoints_flat)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        endpoints_in = x[:, self.endpoints].clone()
        x = x.reshape(-1, 21 * 3)
        x = self.linear1(x)
        x = self.linear2(self.actvn(x))
        x = self.linear3(self.actvn(x))

        y_pred = x.reshape(-1, 21, 3)
        if self.fix_root:
            y_pred[:, self.endpoints] = endpoints_in
        return y_pred


def get_projection_layer(model_path):
    model = torch.load(model_path)
    return model
