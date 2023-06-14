import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleDecoder(nn.Module):
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

        def make_sequence():
            return []

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
    

class PiecewiseRigidDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        num_bones=16,
        projection=None,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        smooth_max=False,
        use_sigmoid=False,
        latent_dropout=False,
    ):
        super(PiecewiseRigidDecoder, self).__init__()

        def make_sequence():
            return []

        if projection is not None:
            dims = [3] + dims + [1]
        else:
            dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.num_bones = num_bones
        self.projection = projection
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.weight_norm = weight_norm

        for bone in range(self.num_bones):
            for layer in range(0, self.num_layers - 1):
                # if layer + 1 in latent_in:
                #     out_dim = dims[layer + 1] - dims[0]
                # else:
                #     out_dim = dims[layer + 1]
                if layer in latent_in:
                    in_dim = dims[layer] + dims[0]
                else:
                    in_dim = dims[layer]
                out_dim = dims[layer + 1]

                # print(in_dim, out_dim)
                if weight_norm and layer in self.norm_layers:
                    setattr(
                        self,
                        "lin" + str(bone) + "_" + str(layer),
                        nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                        # nn.utils.weight_norm(nn.Conv1d(dims[layer], out_dim, 1)),
                    )
                else:
                    setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Linear(in_dim, out_dim))
                    # setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Conv1d(dims[layer], out_dim, 1))

                if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
                ):
                    setattr(self, "bn"  + str(bone) + "_" + str(layer), nn.LayerNorm(out_dim))

        self.smooth_max = smooth_max
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, xyz, latent, reduce_part=True):
        batch_size = xyz.size(0)
        # print("xyz", xyz)

        if self.projection == None:
            # [batch_size, 16, 4, 4] -> [batch_size, 16, tranMat(16)]
            latent = latent.view(batch_size, latent.size(1), 16)
            # [batch_size, 3] -> [batch_size, joints(16), 3]
            xyz = xyz.unsqueeze(1).expand(-1, latent.size(1), -1)
            # [batch_size, joints(16), xyz(3) + tranMat(16)]
            input = torch.cat([latent, xyz], 2)
        elif self.projection == 'x':
            # concat 1 for homogeneous points. [3] -> [4,1]
            xyz = torch.cat([xyz, torch.ones(batch_size, 1, device=xyz.device)], 1).unsqueeze(-1)
            # print(xyz.size())
            # [batch_size, joints(16), 4, 4] x [batch_size, 1, 4, 1] -> [batch_size, 16, 4, 1]
            input = torch.matmul(latent, xyz.unsqueeze(1))
            input = input[:, :, :3, 0]
            # final input [batch_size, joints(16), projection(3)]

        output = torch.zeros([input.size(0), self.num_bones], device=input.device)

        for bone in range(self.num_bones):
            input_i = input[:, bone, :]
            x = input[:, bone, :]
            # print('x size', x.size())
            for layer in range(0, self.num_layers - 1):
                x_prev = x
                lin = getattr(self, "lin" + str(bone) + "_" + str(layer))
                if layer in self.latent_in:
                    x = torch.cat([x, input_i], 1)
                # x = lin(x)
                x_out = lin(x)
                # last layer Sigmoid
                # if layer == self.num_layers - 2 and self.use_sigmoid:
                #     x_out = self.sigmoid(x_out)
                if layer < self.num_layers - 2:
                    if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                    ):
                        bn = getattr(self, "bn"  + str(bone) + "_" + str(layer))
                        x_out = bn(x_out)
                    x_out = self.relu(x_out)
                    if self.dropout is not None and layer in self.dropout:
                        x_out = F.dropout(x_out, p=self.dropout_prob, training=self.training)
                    
                    # residual connection
                    if layer > 0:
                        x_out = x_out + x_prev
                x = x_out

            # if hasattr(self, "th"):
            #     x = self.th(x)
            output[:, bone] = x[:, 0]
        if self.smooth_max and reduce_part:
            # print("before", output.size())
            output = output.logsumexp(1, keepdim=True)
            # print("after", output.size())
        # Sigmoid
        if self.use_sigmoid:
            output = self.sigmoid(output)

        return output # x


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        Crate a two-layers networks with relu activation.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class PosEncoder(nn.Module):
    '''Module to add positional encoding.'''
    def __init__(self, in_features=3):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class PiecewiseDeformableDecoderPIFu(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        use_bone_length=False,
        bone_latent_size=0,
        num_bones=16,
        projection=None,
        global_projection=None,
        global_pose_projection_size=0,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        smooth_max=False,
        use_sigmoid=False,
        latent_dropout=False,
        combine_final=False,
        positional_encoding=False,
        actv='leakyrelu',
        add_feature_dim=32,
        add_feature_layer_idx=1
    ):
        super(PiecewiseDeformableDecoderPIFu, self).__init__()

        def make_sequence():
            return []

        self.add_feature_dim = add_feature_dim
        self.add_feature_layer_idx = add_feature_layer_idx

        # global pose projection type
        if global_projection == 'o':
            # origin
            latent_size = 3
        elif global_projection is None:
            # no projection
            latent_size = 4 * 4
        
        if positional_encoding:
            self.posi_encoder = PosEncoder()
            xyz_size = 3 + 3 * 2 * 10
        else:
            self.posi_encoder = None
            xyz_size = 3

        # temp!!!
        xyz_size = 3 #+ 256
        
        # global pose sub-space projection
        self.global_pose_projection_size = global_pose_projection_size
        if global_pose_projection_size > 0:
            for i in range(num_bones):
                setattr(self, "global_proj" + str(i), nn.Linear(latent_size * num_bones, global_pose_projection_size))
            dims = [xyz_size + global_pose_projection_size] + dims + [1]
        else:
            # self.global_pose_projection_layer = None
            if projection is not None:
                dims = [xyz_size + latent_size * num_bones] + dims + [1]
            else:
                dims = [xyz_size + latent_size + latent_size * num_bones] + dims + [1]
        
        if use_bone_length:
            dims[0] = dims[0] + 1
            # bone_latent: the latent size of the vector encoding all bone lengths
            self.bone_latent_size = bone_latent_size # 16
            if bone_latent_size > 0:
                dims[0] = dims[0] + self.bone_latent_size
                self.bone_encoder = TwoLayerNet(num_bones, 40, self.bone_latent_size)

        self.use_bone_length = use_bone_length
        self.num_layers = len(dims)
        self.num_bones = num_bones
        self.projection = projection
        self.global_projection = global_projection
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)
        
        # combine final output in the last layer
        self.combine_final = combine_final
        if self.combine_final:
            self.combine_final_layer = nn.Linear(dims[-2] * num_bones, 1)

        self.weight_norm = weight_norm

        # Part model
        for bone in range(self.num_bones):
            for layer in range(0, self.num_layers - 1):
                # if layer + 1 in latent_in:
                #     out_dim = dims[layer + 1] - dims[0]
                # else:
                #     out_dim = dims[layer + 1]
                if layer in latent_in:
                    in_dim = dims[layer] + dims[0]
                else:
                    in_dim = dims[layer]

                    if layer == self.add_feature_layer_idx:
                        in_dim += self.add_feature_dim

                out_dim = dims[layer + 1]

                if weight_norm and layer in self.norm_layers:
                    setattr(
                        self,
                        "lin" + str(bone) + "_" + str(layer),
                        nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                        # nn.utils.weight_norm(nn.Conv1d(dims[layer], out_dim, 1)),
                    )
                else:
                    setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Linear(in_dim, out_dim))
                    # setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Conv1d(dims[layer], out_dim, 1))

                if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
                ):
                    setattr(self, "bn"  + str(bone) + "_" + str(layer), nn.LayerNorm(out_dim))
                
                if actv == "siren":
                    if layer == 0:
                        getattr(self, "lin" + str(bone) + "_" + str(layer)).apply(first_layer_sine_init)
                    else:
                        getattr(self, "lin" + str(bone) + "_" + str(layer)).apply(sine_init)

        self.smooth_max = smooth_max
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        if actv == "siren":
            self.actv = Sine()
        else:
            self.actv = nn.LeakyReLU(0.1)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, in_feat, latent, bone_lengths=None, reduce_part=True):
        # print("xyz", xyz, xyz.shape)
        batch_size = in_feat.size(0)

        xyz = in_feat[:, :3]
        img_feat = in_feat[:, 3:]
        
        # print("batch_size", batch_size)

        # Query point projection. Should be (B^-1)(x) by default.
        if self.projection is None:
            # [batch_size, 16, 4, 4] -> [batch_size, 16, tranMat(16)]
            latent_reshape = latent.view(batch_size, latent.size(1), 16)
            # [batch_size, 3] -> [batch_size, joints(16), 3]
            xyz = xyz.unsqueeze(1).expand(-1, latent_reshape.size(1), -1)
            # [batch_size, joints(16), xyz(3) + tranMat(16)]
            input = torch.cat([latent_reshape, xyz], 2)
        elif self.projection == 'x':
            # (B^-1)(x)
            # concat 1 for homogeneous points. [3] -> [4,1]
            xyz = torch.cat([xyz, torch.ones(batch_size, 1, device=xyz.device)], 1).unsqueeze(-1)
            # print(xyz.size())
            # [batch_size, joints(16), 4, 4] x [batch_size, 1, 4, 1] -> [batch_size, 16, 4, 1]
            #latent = torch.eye(4).cuda().unsqueeze(0).unsqueeze(0) # latent is wrong here
            #latent = latent.repeat(xyz.shape[0], 16, 1, 1)
            input = torch.matmul(latent.double(), xyz.unsqueeze(1).double())
            input = input[:, :, :3, 0]
            # final input shape [batch_size, joints(16), projection(3)]
        #elif self.projection == 'o':
        #    # (B^-1)(o)
        #    pass
        
        # Positional encoding
        if self.posi_encoder is not None:
            # import pdb; pdb.set_trace()
            input = self.posi_encoder(input)

        # global latent code projection
        if self.global_projection == 'o':
            # collections of (B^-1)(o)
            global_latent = latent[:, :, :3, 3]
            # print("global_latent", global_latent)
            global_latent = global_latent.reshape(batch_size, -1)
            # print("global size", global_latent.size())
        else:
            # no projection, just flatten the transformation matrix
            global_latent = latent.reshape(batch_size, -1)
        
        # global latent code sub-space projection
        # if self.global_pose_projection_layer:
        #     global_latent = self.global_pose_projection_layer(global_latent)

        # Compute global bone length encoding
        if self.use_bone_length and self.bone_latent_size > 0:
            # print("bone length shape", bone_lengths.shape)
            bone_latent = self.bone_encoder(bone_lengths.float())
            # print("bone latent shape", bone_latent.shape)

        output = torch.zeros([input.size(0), self.num_bones], device=input.device)
        # For combining final latent
        last_layer_latents = []

        ## Input to each sub model is [x; (local bone length); (global bone length latent); global latent]
        # Input to each sub model is [x(3, fixed); local bone length(1, fixed); global bone length latent(16); global latent(8)]
        for bone in range(self.num_bones):
            input_i = input[:, bone, :]
            # print("input shape", input.shape)
            x = input[:, bone, :]
            # print("x shape", x.shape)
            # concat bone length
            # print("bone length shape", bone_lengths.shape)
            # print("bone length shape", bone_lengths[:, bone].shape)
            if self.use_bone_length:
                x = torch.cat([x, bone_lengths[:, bone].unsqueeze(-1)], axis=1)
                if self.bone_latent_size > 0:
                    #x = torch.cat([x, bone_latent, img_feat], axis=1) # this is modified!!!
                    x = torch.cat([x, bone_latent], axis=1) # this is modified!!!
                    # print("x after global bone latent", x.shape)

            # print('x size', x.size())
            # print('global latent', global_latent.size())
            # Per-bone global subspace projection
            if self.global_pose_projection_size > 0:
                global_proj = getattr(self, "global_proj" + str(bone))
                # print("global latent code size", global_latent.size())
                projected_global_latent = global_proj(global_latent)
                x = torch.cat([x, projected_global_latent], 1)
            else:
                x = torch.cat([x, global_latent], 1)

            # print('x before model', x.shape)
            for idx, layer in enumerate(range(0, self.num_layers - 1)):
                x_prev = x
                lin = getattr(self, "lin" + str(bone) + "_" + str(layer))
                if layer in self.latent_in:
                    x = torch.cat([x, input_i], 1)
                
                if idx == self.add_feature_layer_idx:
                    x = torch.cat([x, img_feat], 1)
                
                if layer == self.num_layers - 2 and self.combine_final:
                    last_layer_latents.append(x)
                    # print(x.shape)
                # x = lin(x)
                x_out = lin(x.float())

                # last layer
                # if layer == self.num_layers - 2:
                    # Smooth max log-sum-exp

                if layer < self.num_layers - 2:
                    # residual connection
                    if layer > 0:
                        x_out = x_out + x_prev

                    if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                    ):
                        bn = getattr(self, "bn"  + str(bone) + "_" + str(layer))
                        x_out = bn(x_out)
                    x_out = self.actv(x_out)
                    if self.dropout is not None and layer in self.dropout:
                        x_out = F.dropout(x_out, p=self.dropout_prob, training=self.training)

                x = x_out
            # if hasattr(self, "th"):
            #     x = self.th(x)
            # print("x_out", x.size())
            output[:, bone] = x[:, 0]
        
        if self.combine_final:
            # import pdb; pdb.set_trace()
            print('final')
            output = self.combine_final_layer(torch.cat(last_layer_latents, dim=-1))

        # import pdb; pdb.set_trace()
        # print("output shape", output.size())
        # if self.smooth_max and reduce_part:
        #     print("before", output.size())
        #     output = output.logsumexp(1, keepdim=True)
        #     print("after", output.size())

        # # Sigmoid
        # if self.use_sigmoid:
        #     print("sigmoid")
        #     output = self.sigmoid(output)

        # output = nn.Softmax(dim=1)(output)
        # print(output[0])
        # print(output)
        # output, _ = output.max(1, keepdim=True)
        # print("----output", output.size())
        # print("-------", x.size())

        return output # x


class PiecewiseDeformableDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        use_bone_length=False,
        bone_latent_size=0,
        num_bones=16,
        projection=None,
        global_projection=None,
        global_pose_projection_size=0,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        smooth_max=False,
        use_sigmoid=False,
        latent_dropout=False,
        combine_final=False,
        positional_encoding=False,
        actv='leakyrelu',
    ):
        super(PiecewiseDeformableDecoder, self).__init__()

        def make_sequence():
            return []

        # global pose projection type
        if global_projection == 'o':
            # origin
            latent_size = 3
        elif global_projection is None:
            # no projection
            latent_size = 4 * 4
        
        if positional_encoding:
            self.posi_encoder = PosEncoder()
            xyz_size = 3 + 3 * 2 * 10
        else:
            self.posi_encoder = None
            xyz_size = 3
        
        # global pose sub-space projection
        self.global_pose_projection_size = global_pose_projection_size
        if global_pose_projection_size > 0:
            for i in range(num_bones):
                setattr(self, "global_proj" + str(i), nn.Linear(latent_size * num_bones, global_pose_projection_size))
            dims = [xyz_size + global_pose_projection_size] + dims + [1]
        else:
            # self.global_pose_projection_layer = None
            if projection is not None:
                dims = [xyz_size + latent_size * num_bones] + dims + [1]
            else:
                dims = [xyz_size + latent_size + latent_size * num_bones] + dims + [1]
        
        if use_bone_length:
            dims[0] = dims[0] + 1
            # bone_latent: the latent size of the vector encoding all bone lengths
            self.bone_latent_size = bone_latent_size # 16
            if bone_latent_size > 0:
                dims[0] = dims[0] + self.bone_latent_size
                self.bone_encoder = TwoLayerNet(num_bones, 40, self.bone_latent_size)

        self.use_bone_length = use_bone_length
        self.num_layers = len(dims)
        self.num_bones = num_bones
        self.projection = projection
        self.global_projection = global_projection
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)
        
        # combine final output in the last layer
        self.combine_final = combine_final
        if self.combine_final:
            self.combine_final_layer = nn.Linear(dims[-2] * num_bones, 1)

        self.weight_norm = weight_norm

        # Part model
        for bone in range(self.num_bones):
            for layer in range(0, self.num_layers - 1):
                # if layer + 1 in latent_in:
                #     out_dim = dims[layer + 1] - dims[0]
                # else:
                #     out_dim = dims[layer + 1]
                if layer in latent_in:
                    in_dim = dims[layer] + dims[0]
                else:
                    in_dim = dims[layer]
                out_dim = dims[layer + 1]

                # print(in_dim, out_dim)
                if weight_norm and layer in self.norm_layers:
                    setattr(
                        self,
                        "lin" + str(bone) + "_" + str(layer),
                        nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                        # nn.utils.weight_norm(nn.Conv1d(dims[layer], out_dim, 1)),
                    )
                else:
                    setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Linear(in_dim, out_dim))
                    # setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Conv1d(dims[layer], out_dim, 1))

                if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
                ):
                    setattr(self, "bn"  + str(bone) + "_" + str(layer), nn.LayerNorm(out_dim))
                
                if actv == "siren":
                    if layer == 0:
                        getattr(self, "lin" + str(bone) + "_" + str(layer)).apply(first_layer_sine_init)
                    else:
                        getattr(self, "lin" + str(bone) + "_" + str(layer)).apply(sine_init)

        self.smooth_max = smooth_max
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        if actv == "siren":
            self.actv = Sine()
        else:
            self.actv = nn.LeakyReLU(0.1)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, xyz, latent, bone_lengths=None, reduce_part=True):
        # print("xyz", xyz, xyz.shape)
        batch_size = xyz.size(0)
        # print("batch_size", batch_size)

        # Query point projection. Should be (B^-1)(x) by default.
        if self.projection is None:
            # [batch_size, 16, 4, 4] -> [batch_size, 16, tranMat(16)]
            latent_reshape = latent.view(batch_size, latent.size(1), 16)
            # [batch_size, 3] -> [batch_size, joints(16), 3]
            xyz = xyz.unsqueeze(1).expand(-1, latent_reshape.size(1), -1)
            # [batch_size, joints(16), xyz(3) + tranMat(16)]
            input = torch.cat([latent_reshape, xyz], 2)
        elif self.projection == 'x':
            # (B^-1)(x)
            # concat 1 for homogeneous points. [3] -> [4,1]
            xyz = torch.cat([xyz, torch.ones(batch_size, 1, device=xyz.device)], 1).unsqueeze(-1)
            # print(xyz.size())
            # [batch_size, joints(16), 4, 4] x [batch_size, 1, 4, 1] -> [batch_size, 16, 4, 1]
            #latent = torch.eye(4).cuda().unsqueeze(0).unsqueeze(0) # latent is wrong here
            #latent = latent.repeat(xyz.shape[0], 16, 1, 1)
            input = torch.matmul(latent.double(), xyz.unsqueeze(1).double())
            input = input[:, :, :3, 0]
            # final input shape [batch_size, joints(16), projection(3)]
        #elif self.projection == 'o':
        #    # (B^-1)(o)
        #    pass
        
        # Positional encoding
        if self.posi_encoder is not None:
            # import pdb; pdb.set_trace()
            input = self.posi_encoder(input)

        # global latent code projection
        if self.global_projection == 'o':
            # collections of (B^-1)(o)
            global_latent = latent[:, :, :3, 3]
            # print("global_latent", global_latent)
            global_latent = global_latent.reshape(batch_size, -1)
            # print("global size", global_latent.size())
        else:
            # no projection, just flatten the transformation matrix
            global_latent = latent.reshape(batch_size, -1)
        
        # global latent code sub-space projection
        # if self.global_pose_projection_layer:
        #     global_latent = self.global_pose_projection_layer(global_latent)

        # Compute global bone length encoding
        if self.use_bone_length and self.bone_latent_size > 0:
            # print("bone length shape", bone_lengths.shape)
            bone_latent = self.bone_encoder(bone_lengths.float())
            # print("bone latent shape", bone_latent.shape)


        output = torch.zeros([input.size(0), self.num_bones], device=input.device)
        # For combining final latent
        last_layer_latents = []

        ## Input to each sub model is [x; (local bone length); (global bone length latent); global latent]
        # Input to each sub model is [x(3, fixed); local bone length(1, fixed); global bone length latent(16); global latent(8)]
        for bone in range(self.num_bones):
            input_i = input[:, bone, :]
            # print("input shape", input.shape)
            x = input[:, bone, :]
            # print("x shape", x.shape)
            # concat bone length
            # print("bone length shape", bone_lengths.shape)
            # print("bone length shape", bone_lengths[:, bone].shape)
            if self.use_bone_length:
                x = torch.cat([x, bone_lengths[:, bone].unsqueeze(-1)], axis=1)
                if self.bone_latent_size > 0:
                    x = torch.cat([x, bone_latent], axis=1)
                    # print("x after global bone latent", x.shape)

            # print('x size', x.size())
            # print('global latent', global_latent.size())
            # Per-bone global subspace projection
            if self.global_pose_projection_size > 0:
                global_proj = getattr(self, "global_proj" + str(bone))
                # print("global latent code size", global_latent.size())
                projected_global_latent = global_proj(global_latent)
                x = torch.cat([x, projected_global_latent], 1)
            else:
                x = torch.cat([x, global_latent], 1)
            
            
            # print('x before model', x.shape)
            for layer in range(0, self.num_layers - 1):
                x_prev = x
                lin = getattr(self, "lin" + str(bone) + "_" + str(layer))
                if layer in self.latent_in:
                    x = torch.cat([x, input_i], 1)
                
                if layer == self.num_layers - 2 and self.combine_final:
                    last_layer_latents.append(x)
                    # print(x.shape)
                # x = lin(x)
                x_out = lin(x.float())

                # last layer
                # if layer == self.num_layers - 2:
                    # Smooth max log-sum-exp

                if layer < self.num_layers - 2:
                    # residual connection
                    if layer > 0:
                        x_out = x_out + x_prev

                    if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                    ):
                        bn = getattr(self, "bn"  + str(bone) + "_" + str(layer))
                        x_out = bn(x_out)
                    x_out = self.actv(x_out)
                    if self.dropout is not None and layer in self.dropout:
                        x_out = F.dropout(x_out, p=self.dropout_prob, training=self.training)

                x = x_out

            # if hasattr(self, "th"):
            #     x = self.th(x)
            # print("x_out", x.size())
            output[:, bone] = x[:, 0]
        
        if self.combine_final:
            # import pdb; pdb.set_trace()
            output = self.combine_final_layer(torch.cat(last_layer_latents, dim=-1))

        # import pdb; pdb.set_trace()
        # print("output shape", output.size())
        # if self.smooth_max and reduce_part:
        #     print("before", output.size())
        #     output = output.logsumexp(1, keepdim=True)
        #     print("after", output.size())

        # # Sigmoid
        # if self.use_sigmoid:
        #     print("sigmoid")
        #     output = self.sigmoid(output)

        # output = nn.Softmax(dim=1)(output)
        # print(output[0])
        # print(output)
        # output, _ = output.max(1, keepdim=True)
        # print("----output", output.size())
        # print("-------", x.size())

        return output # x


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class SdfDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        use_bone_length=False,
        bone_latent_size=0,
        num_bones=16,
        projection=None,
        global_projection=None,
        global_pose_projection_size=0,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        smooth_max=False,
        use_sigmoid=False,
        latent_dropout=False,
        combine_final=False,
        positional_encoding=False,
        actv='leakyrelu',
    ):
        super(SdfDecoder, self).__init__()

        # global pose projection type
        if global_projection == 'o':
            # origin
            latent_size = 3
        elif global_projection is None:
            # no projection
            latent_size = 4 * 4
        
        if positional_encoding:
            self.posi_encoder = PosEncoder()
            xyz_size = 3 + 3 * 2 * 10
        else:
            self.posi_encoder = None
            xyz_size = 3
        
        # global pose sub-space projection
        self.global_pose_projection_size = global_pose_projection_size
        if global_pose_projection_size > 0:
            for i in range(num_bones):
                setattr(self, "global_proj" + str(i), nn.Linear(latent_size * num_bones, global_pose_projection_size))
            # dims = [xyz_size + global_pose_projection_size] + dims + [1]
            dims = [(xyz_size + global_pose_projection_size) * num_bones] + dims + [1]
            # print("use global projection")
        else:
            # self.global_pose_projection_layer = None
            if projection is not None:
                dims = [xyz_size + latent_size * num_bones] + dims + [1]
            else:
                dims = [xyz_size + latent_size + latent_size * num_bones] + dims + [1]
        
        # print("dims", dims)
        
        if use_bone_length:
            # dims[0] = dims[0] + 1
            # bone_latent: the latent size of the vector encoding all bone lengths
            self.bone_latent_size = bone_latent_size # 16
            if bone_latent_size > 0:
                dims[0] = dims[0] + self.bone_latent_size
                self.bone_encoder = TwoLayerNet(num_bones, 40, self.bone_latent_size)

        self.use_bone_length = use_bone_length
        self.num_layers = len(dims)
        self.num_bones = num_bones
        self.projection = projection
        self.global_projection = global_projection
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            # if layer + 1 in latent_in:
            #     out_dim = dims[layer + 1] - dims[0]
            # else:
            #     out_dim = dims[layer + 1]
            if layer in latent_in:
                in_dim = dims[layer] + dims[0]
            else:
                in_dim = dims[layer]
            out_dim = dims[layer + 1]

            # print(in_dim, out_dim)
            setattr(self, "lin" + "_" + str(layer), nn.Linear(in_dim, out_dim))
        # setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Conv1d(dims[layer], out_dim, 1))


        self.smooth_max = smooth_max
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        if actv == "siren":
            self.actv = Sine()
        else:
            self.actv = nn.LeakyReLU(0.1)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, xyz, latent, bone_lengths=None, reduce_part=True):
        # print("xyz", xyz, xyz.shape)
        batch_size = xyz.size(0)
        # print("batch_size", batch_size)
        
        # Query point projection. Should be (B^-1)(x) by default.
        if self.projection is None:
            # [batch_size, 16, 4, 4] -> [batch_size, 16, tranMat(16)]
            latent_reshape = latent.view(batch_size, latent.size(1), 16)
            # [batch_size, 3] -> [batch_size, joints(16), 3]
            xyz = xyz.unsqueeze(1).expand(-1, latent_reshape.size(1), -1)
            # [batch_size, joints(16), xyz(3) + tranMat(16)]
            input = torch.cat([latent_reshape, xyz], 2)
        elif self.projection == 'x':
            # (B^-1)(x)
            # concat 1 for homogeneous points. [3] -> [4,1]
            xyz = torch.cat([xyz, torch.ones(batch_size, 1, device=xyz.device)], 1).unsqueeze(-1)
            # print(xyz.size())
            # [batch_size, joints(16), 4, 4] x [batch_size, 1, 4, 1] -> [batch_size, 16, 4, 1]
            input = torch.matmul(latent, xyz.unsqueeze(1))
            input = input[:, :, :3, 0]
            # final input shape [batch_size, joints(16), projection(3)]
        
        # Positional encoding
        if self.posi_encoder is not None:
            # import pdb; pdb.set_trace()
            input = self.posi_encoder(input)

        # global latent code projection
        if self.global_projection == 'o':
            # collections of (B^-1)(o)
            global_latent = latent[:, :, :3, 3]
            # print("global_latent", global_latent)
            global_latent = global_latent.reshape(batch_size, -1)
            # print("global size", global_latent.size())
        else:
            # no projection, just flatten the transformation matrix
            global_latent = latent.reshape(batch_size, -1)
        
        # global latent code sub-space projection
        # if self.global_pose_projection_layer:
        #     global_latent = self.global_pose_projection_layer(global_latent)

        # Compute global bone length encoding
        if self.use_bone_length and self.bone_latent_size > 0:
            # print("bone length shape", bone_lengths.shape)
            bone_latent = self.bone_encoder(bone_lengths)
            # print("bone latent shape", bone_latent.shape)

        output = torch.zeros([input.size(0), self.num_bones], device=input.device)
        bone_input_list = []

        ## Input to each sub model is [x; (local bone length); (global bone length latent); global latent]
        # Input to each sub model is [x(3, fixed); local bone length(1, fixed); global bone length latent(16); global latent(8)]
        for bone in range(self.num_bones):
            input_i = input[:, bone, :]
            # print("input shape", input.shape)
            x = input[:, bone, :]
            # print("x shape", x.shape)
            # concat bone length
            # print("bone length shape", bone_lengths.shape)
            # print("bone length shape", bone_lengths[:, bone].shape)

            # print('x size', x.size())
            # print('global latent', global_latent.size())
            # Per-bone global subspace projection
            if self.global_pose_projection_size > 0:
                global_proj = getattr(self, "global_proj" + str(bone))
                # print("global latent code size", global_latent.size())
                projected_global_latent = global_proj(global_latent)
                x = torch.cat([x, projected_global_latent], 1)
            else:
                x = torch.cat([x, global_latent], 1)

            bone_input_list.append(x)
        
        # import pdb; pdb.set_trace()
        x = torch.cat(bone_input_list, 1)
        if self.use_bone_length:
            # x = torch.cat([x, bone_lengths[:, bone].unsqueeze(-1)], axis=1)
            if self.bone_latent_size > 0:
                x = torch.cat([x, bone_latent], axis=1)
                # print("x after global bone latent", x.shape)
            
            
        # print('x before model', x.shape)
        for layer in range(0, self.num_layers - 1):
            x_prev = x
            lin = getattr(self, "lin" + "_" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input_i], 1)

            x_out = lin(x)

            # last layer
            # if layer == self.num_layers - 2:
                # Smooth max log-sum-exp

            if layer < self.num_layers - 2:
                # residual connection
                if layer > 0:
                    x_out = x_out + x_prev
                x_out = self.actv(x_out)
                if self.dropout is not None and layer in self.dropout:
                    x_out = F.dropout(x_out, p=self.dropout_prob, training=self.training)

            x = x_out
        # import pdb; pdb.set_trace()
        # # if hasattr(self, "th"):
        # #     x = self.th(x)
        # # print("x_out", x.size())
        # output[:, bone] = x[:, 0]

        # import pdb; pdb.set_trace()
        # print("output shape", output.size())
        # if self.smooth_max and reduce_part:
        #     print("before", output.size())
        #     output = output.logsumexp(1, keepdim=True)
        #     print("after", output.size())

        # # Sigmoid
        # if self.use_sigmoid:
        #     print("sigmoid")
        #     output = self.sigmoid(output)

        # output = nn.Softmax(dim=1)(output)
        # print(output[0])
        # print(output)
        # output, _ = output.max(1, keepdim=True)
        # print("----output", output.size())
        # print("-------", x.size())

        return x_out  # output # x
