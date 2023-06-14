import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleDecoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, mano_params_out=False):
        """
        Crate a simple feed-forward networks with relu activation.
        """
        super(SimpleDecoder, self).__init__()
        # self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear2 = torch.nn.Linear(H, H)
        # self.out_layer = torch.nn.Linear(H, D_out)
        self.mano_params_out = mano_params_out
        # print("  ------- mano_params_out", mano_params_out)

        self.fc_1 = torch.nn.Linear(D_in, H)
        self.fc_2 = torch.nn.Linear(H, H)
        self.fc_3 = torch.nn.Linear(H, H)
        self.fc_4 = torch.nn.Linear(H, D_out)

        self.actvn = nn.LeakyReLU(0.1)

    def forward(self, z, c):
        x = torch.cat([z, c], 1)
        # h_relu = self.linear1(x).clamp(min=0)
        # h2_relu = self.linear2(h_relu).clamp(min=0)
        # y_pred = self.out_layer(h2_relu)

        net = self.fc_1(x)
        net = self.fc_2(self.actvn(net)) + net
        net = self.fc_3(self.actvn(net)) + net
        net = self.fc_4(self.actvn(net))

        # mano_params_out = True
        if self.mano_params_out:
            y_pred = net
        else:
            y_pred = net.reshape(-1, 21, 3)
        # y_pred = net.reshape(-1, 21, 3)

        # print("y_pred", y_pred.shape)
        # y_pred = y_pred.reshape(-1, 21, 3)
        return y_pred


class SimpleDecoderss(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        # xyz_in_all=None,
        use_sigmoid=False,
        latent_dropout=False,
    ):
        super(SimpleDecoder, self).__init__()

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        # self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                # if self.xyz_in_all and layer != self.num_layers - 2:
                #     out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))
            # print(dims[layer], out_dim)

        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, xyz, latent, reduce_part=False):
        batch_size = xyz.size(0)
        # print("latent size", latent.size())
        # print(latent)
        # reshape from [batch_size, 16, 4, 4] to [batch_size, 256]
        latent = latent.reshape(batch_size, -1)
        # print("latent size", latent.size())
        # print(latent)
        # print("xyz size", xyz.size())
        input = torch.cat([latent, xyz], 1)
        x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            # elif layer != 0 and self.xyz_in_all:
            #     x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Sigmoid
            if layer == self.num_layers - 2 and self.use_sigmoid:
                x = self.sigmoid(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # if hasattr(self, "th"):
        #     x = self.th(x)

        return x
    