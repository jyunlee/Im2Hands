'''
AIR-Nets
Author: Simon Giebenhain
Code: https://github.com/SimonGiebenhain/AIR-Nets
'''

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from time import time
import torch.nn.functional as F
import os
import math
import dependencies.airnets.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils


def fibonacci_sphere(samples=1):
    '''
    Code from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    Args:
        samples: number of samples

    Returns:
        Points evenly distributed on the unit sphere
    '''
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append(np.array([x, y, z]))

    return np.stack(points, axis=0)


def square_distance(src, dst):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py

    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class TransitionDown(nn.Module):
    """
        High-level wrapper for different downsampling mechanisms (also called set abstraction mechanisms).
        In general the point cloud is subsampled to produce a lower cardinality point cloud (usualy using farthest point
        sampling (FPS) ). Around each of the resulting points (called central points here) a local neighborhood is
        formed, from which features are aggregated. How features are aggregated can differ, usually this is based on
        maxpooling. This work introduces an attention based alternative.

        Attributes:
            npoint: desired number of points for outpout point cloud
            nneigh: size of neighborhood
            dim: number of dimensions of input and interal dimensions
            type: decides which method to use, options are 'attentive' and 'maxpool'
        """
    def __init__(self, npoint, nneighbor, dim, type='attentive') -> None:
        super().__init__()
        if type == 'attentive':
            self.sa = TransformerSetAbstraction(npoint, nneighbor, dim)
        elif type == 'maxpool':
            self.sa = PointNetSetAbstraction(npoint, nneighbor, dim, dim)
        else:
            raise ValueError('Set Abstraction type ' + type + ' unknown!')

    def forward(self, xyz, feats):
        """
        Executes the downsampling (set abstraction)
        :param xyz: positions of points
        :param feats: features of points
        :return: downsampled version, tuple of (xyz_new, feats_new)
        """
        ret = self.sa(xyz, feats)
        return ret


class TransformerBlock(nn.Module):
    """
    Module for local and global vector self attention, as proposed in the Point Transformer paper.

    Attributes:
        d_model (int): number of input, output and internal dimensions
        k (int): number of points among which local attention is calculated
        pos_only (bool): When set to True only positional features are used
        group_all (bool): When true full instead of local attention is calculated
    """
    def __init__(self, d_model, k, pos_only=False, group_all=False) -> None:
        super().__init__()

        self.pos_only = pos_only

        self.bn = nn.BatchNorm1d(d_model)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.group_all = group_all

    def forward(self, xyz, feats=None):
        """
        :param xyz [b x n x 3]: positions in point cloud
        :param feats [b x n x d]: features in point cloud
        :return:
            new_features [b x n x d]:
        """

        with torch.no_grad():
            # full attention
            if self.group_all:
                b, n, _ = xyz.shape
                knn_idx = torch.arange(n, device=xyz.device).unsqueeze(0).unsqueeze(1).repeat(b, n, 1)
            # local attention using KNN
            else:
                dists = square_distance(xyz, xyz)
                knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k

        knn_xyz = index_points(xyz, knn_idx)

        if not self.pos_only:
            ori_feats = feats
            x = feats

            q_attn = self.w_qs(x)
            k_attn = index_points(self.w_ks(x), knn_idx)
            v_attn = index_points(self.w_vs(x), knn_idx)

        pos_encode = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d

        if not self.pos_only:
            attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        else:
            attn = self.fc_gamma(pos_encode)


        attn = functional.softmax(attn, dim=-2)  # b x n x k x d
        if not self.pos_only:
            res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)
        else:
            res = torch.einsum('bmnf,bmnf->bmf', attn, pos_encode)



        if not self.pos_only:
            res = res + ori_feats
        res = self.bn(res.permute(0, 2, 1)).permute(0, 2, 1)

        return res


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_inp, dim, nneigh=7, reduce_dim=True, separate_delta=True):
        super().__init__()

        # dim_inp = dim
        # dim = dim  # // 2
        self.dim = dim

        self.nneigh = nneigh
        self.separate_delta = separate_delta

        self.fc_delta = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        #if self.separate_delta:
        #    self.fc_delta2 = nn.Sequential(
        #        nn.Linear(3, dim),
        #        nn.ReLU(),
        #        nn.Linear(dim, dim)
        #
        #    )

        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.w_k_global = nn.Linear(dim_inp, dim, bias=False)
        self.w_v_global = nn.Linear(dim_inp, dim, bias=False)

        self.w_qs = nn.Linear(dim_inp, dim, bias=False)
        self.w_ks = nn.Linear(dim_inp, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp, dim, bias=False)

        if not reduce_dim:
            self.fc = nn.Linear(dim, dim_inp)
        self.reduce_dim = reduce_dim

    # xyz_q: B x n_queries x 3
    # lat_rep: B x dim
    # xyz: B x n_anchors x 3,
    # points: B x n_anchors x dim
    def forward(self, xyz_q, lat_rep, xyz, points):
        with torch.no_grad():
            dists = square_distance(xyz_q, xyz)
            ## knn group
            knn_idx = dists.argsort()[:, :, :self.nneigh]  # b x nQ x k
            #print(knn_idx.shape)

            #knn = KNN(k=self.nneigh, transpose_mode=True)
            #_, knn_idx = knn(xyz, xyz_q)  # B x npoint x K
            ##
            #print(knn_idx.shape)

        b, nQ, _ = xyz_q.shape
        # b, nK, dim = points.shape

        if len(lat_rep.shape) == 2:
            q_attn = self.w_qs(lat_rep).unsqueeze(1).repeat(1, nQ, 1)
            k_global = self.w_k_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
        else:
            q_attn = self.w_qs(lat_rep)
            k_global = self.w_k_global(lat_rep).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(2)

        k_attn = index_points(self.w_ks(points),
                              knn_idx)  # b, nQ, k, dim  # self.w_ks(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        k_attn = torch.cat([k_attn, k_global], dim=2)
        v_attn = index_points(self.w_vs(points), knn_idx)  # #self.w_vs(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        v_attn = torch.cat([v_attn, v_global], dim=2)
        xyz = index_points(xyz, knn_idx)  # xyz = xyz.unsqueeze(1).repeat(1, nQ, 1, 1)
        pos_encode = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
        pos_encode = torch.cat([pos_encode, torch.zeros([b, nQ, 1, self.dim], device=pos_encode.device)],
                               dim=2)  # b, nQ, k+1, dim
        if self.separate_delta:
            pos_encode2 = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
            pos_encode2 = torch.cat([pos_encode2, torch.zeros([b, nQ, 1, self.dim], device=pos_encode2.device)],
                                   dim=2)  # b, nQ, k+1, dim
        else:
            pos_encode2 = pos_encode

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x nQ x k+1 x dim

        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode2)  # b x nQ x dim

        if not self.reduce_dim:
            res = self.fc(res)
        return res


class ElementwiseMLP(nn.Module):
    """
    Simple MLP, consisting of two linear layers, a skip connection and batch norm.
    More specifically: linear -> BN -> ReLU -> linear -> BN -> ReLU -> resCon -> BN

    Sorry for that many norm layers. I'm sure not all are needed!
    At some point it was just too late to change it to something proper!
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, x):
        """
        :param x: [B x n x d]
        :return: [B x n x d]
        """
        x = x.permute(0, 2, 1)
        return self.bn3(x + F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))).permute(0, 2, 1)


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Copied from https://github.com/autonomousvision/convolutional_occupancy_networks

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


class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Module, as used in PointNet++
    Uses FPS for downsampling, kNN groupings and maxpooling to abstract the group/neighborhood

    Attributes:
        npoint (int): Output cardinality
        nneigh (int): Size of local grouings/neighborhoods
        in_channel (int): input dimensionality
        dim (int): internal and output dimensionality
    """
    def __init__(self, npoint, nneigh, in_channel, dim):
        super(PointNetSetAbstraction, self).__init__()

        self.npoint = npoint
        self.nneigh = nneigh
        self.fc1 = nn.Linear(in_channel, dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.bn = nn.BatchNorm1d(dim)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """

        with torch.no_grad():
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint).long()

        new_xyz = index_points(xyz, fps_idx)
        points = self.fc1(points)
        points_ori = index_points(points, fps_idx)

        points = points.permute(0, 2, 1)
        points = points + F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(points))))))
        points = points.permute(0, 2, 1)

        with torch.no_grad():
            dists = square_distance(new_xyz, xyz)  # B x npoint x N
            idx = dists.argsort()[:, :, :self.nneigh]  # B x npoint x K


        grouped_points = index_points(points, idx)


        new_points = points_ori + torch.max(grouped_points, 2)[0]
        new_points = self.bn(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        return new_xyz, new_points


#TODO: can I share some code with PTB??
class TransformerSetAbstraction(nn.Module):
    """
    Newly proposed attention based set abstraction module.
    Uses cross attention from central point to its neighbors instead of maxpooling.

    Attributes:
        npoint (int): Output cardinality of point cloud
        nneigh (int): size of neighborhoods
        dim (int): input, internal and output dimensionality
    """
    def __init__(self, npoint, nneigh, dim):
        super(TransformerSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nneigh = nneigh

        self.bnorm0 = nn.BatchNorm1d(dim)
        self.bnorm1 = nn.BatchNorm1d(dim)
        self.bnorm2 = nn.BatchNorm1d(dim)

        self.bn1 = nn.BatchNorm1d(dim)

        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)

        self.fc_delta1 = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.fc_gamma1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.fc_gamma2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.w_qs = nn.Linear(dim, dim, bias=False)
        self.w_ks = nn.Linear(dim, dim, bias=False)
        self.w_vs = nn.Linear(dim, dim, bias=False)

        self.w_qs2 = nn.Linear(dim, dim, bias=False)
        self.w_ks2 = nn.Linear(dim, dim, bias=False)
        self.w_vs2 = nn.Linear(dim, dim, bias=False)

    def forward(self, xyz, points):
        """
        Input: featureized point clouds of cardinality N
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, dim]
        Return: downsampled point cloud of cardinality npoint
            new_xyz: sampled points position data, [B, npoint, 3]
            new_points_concat: sample points feature data, [B, npoint, dim]
        """

        B, N, C = xyz.shape

        with torch.no_grad():
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx.long())
        with torch.no_grad():
            dists = square_distance(new_xyz, xyz)  # B x npoint x N
            idx = dists.argsort()[:, :, :self.nneigh]  # B x npoint x K

        q_attn = index_points(self.w_qs(points), fps_idx.long())
        k_attn = index_points(self.w_ks(points), idx)
        v_attn = index_points(self.w_vs(points), idx)
        grouped_xyz = index_points(xyz, idx)

        pos_encode = self.fc_delta1(grouped_xyz - new_xyz.view(B, self.npoint, 1, C))  # b x n x k x f
        attn = self.fc_gamma1(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x n x k x f
        res1 = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)

        res1 = res1 + self.conv2(F.relu(self.bn1(self.conv1(res1.permute(0, 2, 1))))).permute(0, 2, 1)
        res1 = self.bnorm0(res1.permute(0, 2, 1)).permute(0, 2, 1)

        q_attn = self.w_qs2(res1)
        k_attn = index_points(self.w_ks2(points), idx)
        v_attn = index_points(self.w_vs2(points), idx)
        attn = self.fc_gamma2(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x n x k x f
        res2 = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)

        new_points = self.bnorm1((res1 + res2).permute(0, 2, 1)).permute(0, 2, 1)
        new_points = new_points + index_points(points, fps_idx.long())
        new_points = self.bnorm2(new_points.permute(0, 2, 1)).permute(0, 2, 1)

        return new_xyz, new_points


class PointTransformerEncoderV2(nn.Module):
    """
    AIR-Net encoder.

    Attributes:
        npoints_per_layer [int]: cardinalities of point clouds for each layer
        nneighbor int: number of neighbors in local vector attention (also in TransformerSetAbstraction)
        nneighbor_reduced int: number of neighbors in very first TransformerBlock
        nfinal_transformers int: number of full attention layers
        d_transformer int: dimensionality of model
        d_reduced int: dimensionality of very first layers
        full_SA bool: if False, local self attention is used in final layers
        has_features bool: True, when input has signed-distance value for each point
    """
    def __init__(self, npoints_per_layer, nneighbor, nneighbor_reduced, nfinal_transformers,
                 d_transformer, d_reduced,
                 full_SA=False, has_features=False):
        super().__init__()
        self.d_reduced = d_reduced
        self.d_transformer = d_transformer
        self.has_features = has_features

        self.fc_middle = nn.Sequential(
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer)
        )
        if self.has_features:
            self.enc_sdf = nn.Linear(32+2, d_reduced)
        self.transformer_begin = TransformerBlock(d_reduced, nneighbor_reduced,
                                                  pos_only=not self.has_features)
        self.transition_downs = nn.ModuleList()
        self.transformer_downs = nn.ModuleList()
        self.elementwise = nn.ModuleList()
        #self.transformer_downs2 = nn.ModuleList() #compensate
        #self.elementwise2 = nn.ModuleList() # compensate
        self.elementwise_extras = nn.ModuleList()

        if not d_reduced == d_transformer:
            self.fc1 = nn.Linear(d_reduced, d_transformer)

        for i in range(len(npoints_per_layer) - 1):
            old_npoints = npoints_per_layer[i]
            new_npoints = npoints_per_layer[i + 1]

            if i == 0:
                dim = d_reduced
            else:
                dim = d_transformer
            self.transition_downs.append(
                TransitionDown(new_npoints, min(nneighbor, old_npoints), dim) # , type='single_step')  #, type='maxpool')#, type='single_step')
            )
            self.elementwise_extras.append(ElementwiseMLP(dim))
            self.transformer_downs.append(
                TransformerBlock(dim, min(nneighbor, new_npoints))
            )
            self.elementwise.append(ElementwiseMLP(d_transformer))
            #self.transformer_downs2.append(
            #    TransformerBlock(dim, min(nneighbor, new_npoints))
            #) # compensate
            #self.elementwise2.append(ElementwiseMLP(dim)) # compensate

        self.final_transformers = nn.ModuleList()
        self.final_elementwise = nn.ModuleList()

        for i in range(nfinal_transformers):
            self.final_transformers.append(
                TransformerBlock(d_transformer, 2 * nneighbor, group_all=full_SA)
            )
        for i in range(nfinal_transformers):
            self.final_elementwise.append(
                ElementwiseMLP(dim=d_transformer)
            )

    def forward(self, xyz, intermediate_out_path=None):
        """
        :param xyz [B x n x 3] (or [B x n x 4], but then has_features=True): input point cloud
        :param intermediate_out_path: path to store point cloud after every deformation to
        :return: global latent representation [b x d_transformer]
                 xyz [B x npoints_per_layer[-1] x d_transformer]: anchor positions
                 feats [B x npoints_per_layer[-1] x d_transformer: local latent vectors
        """

        if intermediate_out_path is not None:
            intermediates = {}
            intermediates['Input'] = xyz[0, :, :].cpu().numpy()

        if self.has_features:
            #print(xyz[:, :, 3:].shape)
            feats = self.enc_sdf(xyz[:, :, 3:])
            xyz = xyz[:, :, :3].contiguous()
            feats = self.transformer_begin(xyz, feats)
        else:
            feats = self.transformer_begin(xyz)

        for i in range(len(self.transition_downs)):
            xyz, feats = self.transition_downs[i](xyz, feats)

            if intermediate_out_path is not None:
                intermediates['SetAbs{}'.format(i)] = xyz[0, :, :].cpu().numpy()

            feats = self.elementwise_extras[i](feats)
            feats = self.transformer_downs[i](xyz, feats)
            if intermediate_out_path is not None:
                intermediates['PTB{}'.format(i)] = xyz[0, :, :].cpu().numpy()
            #feats = self.transformer_downs2[i](xyz, feats) #compensate: dense
            #feats = self.elementwise2[i](feats) #compensate: dense
            if i == 0 and not self.d_reduced == self.d_transformer:
                feats = self.fc1(feats)
            feats = self.elementwise[i](feats)
            #feats = self.transformer_downs2[i](xyz, feats) #compensate: sparse
            #feats = self.elementwise2[i](feats) #compensate: sparse

        for i, att_block in enumerate(self.final_transformers):
            feats = att_block(xyz, feats)
            #if i < len(self.final_elementwise):
            feats = self.final_elementwise[i](feats)
            if intermediate_out_path is not None:
                intermediates['fullPTB{}'.format(i)] = xyz[0, :, :].cpu().numpy()

        if intermediate_out_path is not None:
            if not os.path.exists(intermediate_out_path):
                os.makedirs(intermediate_out_path)
            np.savez(intermediate_out_path + '/intermediate_pcs.npz', **intermediates)

        # max pooling
        lat_vec = feats.max(dim=1)[0]

        return {'z': self.fc_middle(lat_vec), 'anchors': xyz, 'anchor_feats': feats}


class PointNetEncoder(nn.Module):
    """
    PointNet++-style encoder. Used in ablation experiments.

    Attributes:
        npoints_per_layer [int]: cardinality of point cloud for each layer
        nneighbor int: number of neighbors for set abstraction
        d_transformer int: internal dimensions
    """
    def __init__(self, npoints_per_layer, nneighbor, d_transformer, nfinal_transformers):
        super().__init__()
        self.d_transformer = d_transformer

        self.fc_middle = nn.Sequential(
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer)
        )

        self.fc_begin = nn.Sequential(
            nn.Linear(3, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer)
        )

        self.transition_downs = nn.ModuleList()
        self.elementwise = nn.ModuleList()

        for i in range(len(npoints_per_layer) - 1):
            old_npoints = npoints_per_layer[i]
            new_npoints = npoints_per_layer[i + 1]
            self.transition_downs.append(
                TransitionDown(new_npoints, min(nneighbor, old_npoints), d_transformer, type='maxpool')
            )
            self.elementwise.append(ElementwiseMLP(d_transformer))

        # full self attention layers
        self.final_transformers = nn.ModuleList()
        self.final_elementwise = nn.ModuleList()

        for i in range(nfinal_transformers):
            self.final_transformers.append(
                TransformerBlock(d_transformer, -1, group_all=True)
            )
        for i in range(nfinal_transformers):
            self.final_elementwise.append(
                ElementwiseMLP(dim=d_transformer)
            )

    def forward(self, xyz):
        """
        :param xyz [B x n x 3] (or [B x n x 4], but then has_features=True): input point cloud
        :param intermediate_out_path: path to store point cloud after every deformation to
        :return: global latent representation [b x d_transformer]
                 xyz [B x npoints_per_layer[-1] x d_transformer]: anchor positions
                 feats [B x npoints_per_layer[-1] x d_transformer: local latent vectors
        """
        feats = self.fc_begin(xyz)

        for i in range(len(self.transition_downs)):
            xyz, feats = self.transition_downs[i](xyz, feats)
            feats = self.elementwise[i](feats)

        for i, att_block in enumerate(self.final_transformers):
            feats = att_block(xyz, feats)
            feats = self.final_elementwise[i](feats)

        # max pooling
        lat_vec = feats.max(dim=1)[0]

        return {'z': self.fc_middle(lat_vec), 'anchors': xyz, 'anchor_feats': feats}


class PointTransformerDecoderOcc(nn.Module):
    """
    AIR-Net decoder

    Attributes:
        dim_inp int: dimensionality of encoding (global and local latent vectors)
        dim int: internal dimensionality
        nneigh int: number of nearest anchor points to draw information from
        hidden_dim int: hidden dimensionality of final feed-forward network
        n_blocks int: number of blocks in feed forward network
    """
    def __init__(self, dim_inp, dim, nneigh=7, hidden_dim=64, n_blocks=5, return_feature=False):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        self.return_feature = return_feature


        self.ct1 = CrossTransformerBlock(dim_inp, dim, nneigh=nneigh)
        #self.fc_glob = nn.Linear(dim_inp, dim)

        # WARNING! Ablation
        self.init_enc = nn.Linear(dim+64, hidden_dim)
        #self.init_enc = nn.Linear(dim+32, hidden_dim)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, 1)

        self.actvn = F.tanh

    def forward(self, xyz_q, encoding):
        """
        TODO update commont to include encoding dict
        :param xyz_q [B x n_queries x 3]: queried 3D coordinates
        :param lat_rep [B x dim_inp]: global latent vectors
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchros x dim_inp]: local latent vectors
        :return: occ [B x n_queries]: occupancy probability for each queried 3D coordinate
        """

        lat_rep = encoding['z']
        xyz = encoding['anchors']
        feats = encoding['anchor_feats']

        xyz_q, xyz_q_feat = xyz_q[:, :, :3], xyz_q[:, :, 3:]

        lat_rep = self.ct1(xyz_q, lat_rep, xyz, feats)  # + self.fc_glob(lat_rep).unsqueeze(1).repeat(1, xyz_q.shape[1], 1) +

        cat_lat_rep = torch.cat((lat_rep, xyz_q_feat), dim=2)
        net = self.init_enc(cat_lat_rep)

        # incorporate it here
        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)

        if not self.return_feature: 
            occ = self.fc_out(self.actvn(net))
        else:
            #occ = net
            occ = self.fc_out(net)

        return occ


class PointTransformerDecoderInterp(nn.Module):
    """
    Decoder based in interpolation features between local latent vectors.
    Gaussian Kernel regression is used for the interpolation of features.
    Coda adapted from https://github.com/autonomousvision/convolutional_occupancy_networks

    Attributes:
        dim_inp: input dimensionality
        hidden_dim: dimensionality for feed-forward network
        n_blocks: number of blocks in feed worward network
        var (float): variance for gaussian kernel
    """
    def __init__(self, dim_inp, dim, hidden_dim=50, n_blocks=5):
        super().__init__()
        self.n_blocks = n_blocks

        self.fc0 = nn.Linear(dim_inp, dim)

        self.fc1 = nn.Linear(dim, hidden_dim)


        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, 1)

        self.actvn = F.relu

        self.var = 0.2**2


    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        # p, fea = c

        # distance betweeen each query point to the point cloud
        dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2
        weight = (dist / self.var).exp()  # Guassian kernel

        # weight normalization
        weight = weight / weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea  # B x M x c_dim

        return c_out


    def forward(self, xyz_q, encoding):
        """
        :param xyz_q [B x n_quries x 3]: queried 3D positions
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchors x dim_inp]: anchor features
        :return: occ [B x n_queries]: occupancy predictions/probabilites
        """

        xyz = encoding['anchors']
        feats = encoding['anchor_feats']

        lat_rep = self.fc0(self.sample_point_feature(xyz_q, xyz, feats))

        net = self.fc1(F.relu(lat_rep))

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)

        occ = self.fc_out(self.actvn(net))
        return occ


class PointTransformerDecoderLDIF(nn.Module):
    """
    Decoder based in interpolation features between local latent vectors.
    Gaussian Kernel regression is used for the interpolation of features.
    Coda adapted from https://github.com/autonomousvision/convolutional_occupancy_networks

    Attributes:
        dim_inp: input dimensionality
        hidden_dim: dimensionality for feed-forward network
        n_blocks: number of blocks in feed worward network
        var (float): variance for gaussian kernel
    """
    def __init__(self, dim_inp, dim, hidden_dim=50, n_blocks=5):
        super().__init__()
        self.n_blocks = n_blocks

        self.fc_sclae = nn.Linear(dim_inp, 3)
        self.fc_rot = nn.Linear(dim_inp, 3)


        self.fc0 = nn.Linear(dim_inp+3, dim)

        self.fc1 = nn.Linear(dim, hidden_dim)


        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, 1)

        self.actvn = F.relu


    def euler2mat(self, angles):
        x_angle = angles[:, :, 0]
        y_angle = angles[:, :, 1]
        z_angle = angles[:, :, 2]

        cosz = torch.cos(z_angle)
        sinz = torch.sin(z_angle)
        z_rot = torch.zeros_like(z_angle).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)
        z_rot[:, :, 0, 0] = cosz
        z_rot[:, :, 0, 1] = -sinz
        z_rot[:, :, 1, 0] = sinz
        z_rot[:, :, 1, 1] = cosz
        z_rot[:, :, 2, 2] = 1

        cosy = torch.cos(y_angle)
        siny = torch.sin(y_angle)
        y_rot = torch.zeros_like(y_angle).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)
        y_rot[:, :, 0, 0] = cosy
        y_rot[:, :, 0, 2] = siny
        y_rot[:, :, 1, 1] = 1
        y_rot[:, :, 2, 0] = -siny
        y_rot[:, :, 2, 2] = cosy

        rot = torch.matmul(z_rot, y_rot)

        cosx = torch.cos(x_angle)
        sinx = torch.sin(x_angle)
        x_rot = torch.zeros_like(x_angle).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)
        x_rot[:, :, 0, 0] = 1
        x_rot[:, :, 1, 1] = cosx
        x_rot[:, :, 1, 2] = -sinx
        x_rot[:, :, 2, 1] = sinx
        x_rot[:, :, 2, 2] = cosx

        return torch.matmul(rot, x_rot)

    def compute_weight(self, q, p, f):
        # q: B x M x 3
        # p: B x N x 3
        # f: B x N x hidden_dim

        #TODO: think of better activation function?
        scale = torch.eye(3, device=f.device) * (0.005 + F.sigmoid(self.fc_sclae(f)).unsqueeze(-1).repeat(1, 1, 1, 3)) # B x N x 3 x 3
        rot = self.euler2mat(2*math.pi * F.sigmoid(self.fc_rot(f))) # B x N x 3 -> B x N x 3 x 3
        #cov = torch.matmul(scale, rot)
        cov = scale
        cov_inv = torch.inverse(cov)
        cov_det = torch.det(cov)

        #distance betweeen each query point to the point cloud
        delta = p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)
        tmp = torch.matmul(cov_inv.unsqueeze(1), delta.unsqueeze(-1)).squeeze()
        #m = torch.matmul(delta.unsqueeze(-2), tmp).squeeze()
        m = (delta * tmp).sum(-1)
        dist = (-1/2*m).exp()
        weight = (dist / (2*math.pi)**3 * cov_det.unsqueeze(1)) # B x M x N

        # weight normalization
        #weight = weight / weight.sum(dim=2).unsqueeze(-1)

        return weight, delta, rot


    def forward(self, xyz_q, encoding):
        """
        :param xyz_q [B x n_quries x 3]: queried 3D positions
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchors x dim_inp]: anchor features
        :return: occ [B x n_queries]: occupancy predictions/probabilites
        """

        xyz = encoding['anchors']
        feats = encoding['anchor_feats']
        weights, delta, rot = self.compute_weight(xyz_q, xyz, feats) # B x n_q x n_a

        feats = feats.unsqueeze(1).repeat(1, xyz_q.shape[1], 1, 1)
        loc_coord = torch.matmul(rot.unsqueeze(1), delta.unsqueeze(-1)).squeeze()
        #loc_coord = delta.squeeze()
        feats = torch.cat([feats.squeeze(), loc_coord], dim=-1)

        lat_rep = self.fc0(feats)

        net = self.fc1(F.relu(lat_rep))

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)

        occs = self.fc_out(self.actvn(net)).squeeze() # B x n_q x n_a

        occ = (occs * weights).sum(dim=-1)

        return occ


def get_encoder(CFG):
    CFG_enc = CFG['encoder']
    if CFG_enc['type'] == 'airnet':
        encoder = PointTransformerEncoderV2(npoints_per_layer=CFG_enc['npoints_per_layer'],
                                            nneighbor=CFG_enc['encoder_nneigh'],
                                            nneighbor_reduced=CFG_enc['encoder_nneigh_reduced'],
                                            nfinal_transformers=CFG_enc['nfinal_trans'],
                                            d_transformer=CFG_enc['encoder_attn_dim'],
                                            d_reduced=CFG_enc['encoder_attn_dim_reduced'],
                                            full_SA=CFG_enc.get('full_SA', True))
    elif CFG_enc['type'] == 'pointnet++':
        encoder = PointNetEncoder(npoints_per_layer=CFG_enc['npoints_per_layer'],
                                  nneighbor=CFG_enc['encoder_nneigh'],
                                  d_transformer=CFG_enc['encoder_attn_dim'],
                                  nfinal_transformers=CFG_enc['nfinal_transformers'])
    else:
        raise ValueError('Unrecognized encoder type: ' + CFG_enc['type'])
    return encoder


def get_decoder(CFG):
    CFG_enc = CFG['encoder']
    CFG_dec = CFG['decoder']
    if CFG_dec['type'] == 'airnet':
        decoder = PointTransformerDecoderOcc(dim_inp=CFG_enc['encoder_attn_dim'],
                                             dim=CFG_dec['decoder_attn_dim'],
                                             nneigh=CFG_dec['decoder_nneigh'],
                                             hidden_dim=CFG_dec['decoder_hidden_dim'])
    elif CFG_dec['type'] == 'interp':
        print('Using interpolation-based decoder!')
        decoder = PointTransformerDecoderInterp(dim_inp=CFG_enc['encoder_attn_dim'],
                                                dim=CFG_dec['decoder_attn_dim'], #TODO remove this unnecessary param
                                                hidden_dim=CFG_dec['decoder_hidden_dim'])
    elif CFG_dec['type'] == 'ldif':
        print('Using interpolation-based decoder!')
        decoder = PointTransformerDecoderLDIF(dim_inp=CFG_enc['encoder_attn_dim'],
                                                dim=CFG_dec['decoder_attn_dim'], #TODO ??remove this unnecessary param??
                                                hidden_dim=CFG_dec['decoder_hidden_dim'])
    else:
        raise ValueError('Decoder type "{}" not implemented!'.format(CFG_dec['type']))
    return decoder
