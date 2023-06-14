import sys
import torch
import torch.nn as nn
from torch import distributions as dist

from dependencies.halo.halo_adapter.converter import PoseConverter, transform_to_canonical
from dependencies.halo.halo_adapter.interface import (get_halo_model, convert_joints, change_axes, scale_halo_trans_mat)
from dependencies.halo.halo_adapter.projection import get_projection_layer
from dependencies.halo.halo_adapter.transform_utils import xyz_to_xyz1

from dependencies.intaghand.models.encoder import ResNetSimple
from dependencies.intaghand.models.model_attn.img_attn import *
from dependencies.intaghand.models.model_attn.self_attn import *


class ArticulatedHandNetInitOcc(nn.Module):
    ''' Occupancy Network class.'''

    def __init__(self, left_decoder, right_decoder, device=None):

        super().__init__()

        self.image_encoder = ResNetSimple(model_type='resnet50',
                                          pretrained=True,
                                          fmapDim=[128, 128, 128, 128],
                                          handNum=2,
                                          heatmapDim=21)

        self.image_final_layer = nn.Conv2d(256, 32, 1)

        self.left_pt_embeddings = nn.Sequential(nn.Conv1d(3, 32, 1),
                                                nn.ReLU(),
                                                nn.Dropout(0.01),
                                                nn.Conv1d(32, 16, 1))

        self.right_pt_embeddings = nn.Sequential(nn.Conv1d(3, 32, 1),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.01),
                                                 nn.Conv1d(32, 16, 1))

        self.img_ex_left = img_ex(64, 32,        # img_size, img_f_dim
                                  4, 32,         # grid_size, grid_f_dim
                                  19,            # verts_f_dim
                                  n_heads=2,
                                  dropout=0.01)

        self.img_ex_right = img_ex(64, 32,       # img_size, img_f_dim
                                   4, 32,        # grid_size, grid_f_dim
                                   19,           # verts_f_dim
                                   n_heads=2,
                                   dropout=0.01)

        self.left_decoder = left_decoder.to(device)
        self.right_decoder = right_decoder.to(device)

        self._device = device


    def forward(self, img, camera_params, inputs, p, root_rot_mat, bone_lengths, sample=True, pen=False, **kwargs):
        ''' Performs a forward pass through the network.'''
        left_c, right_c = inputs

        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.image_encoder(img.cuda())

        img_feat = torch.cat((hms_fmaps[-1], dp_fmaps[-1]), 1)
        img_feat = self.image_final_layer(img_feat)

        left_p_r, right_p_r = self.decode(img_feat, camera_params, inputs, p, root_rot_mat, bone_lengths, pen=pen, **kwargs)

        return left_p_r, right_p_r


    def decode(self, img_feat, camera_params, c, p, root_rot_mat, bone_lengths, reduce_part=True, return_model_indices=False, test=False, pen=False, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.'''

        for side in ['left', 'right']:
            # swap query points for penetration check during refined occupancy estimation
            if pen:
                if side == 'left':
                    side = 'right'
                else:
                    side = 'left'

            img_p = p[side] * 0.4

            if test:
                img_p = img_p * 0.4

            img_p = torch.bmm(xyz_to_xyz1(img_p), root_rot_mat[side])[:, :, :3]
            img_p = img_p + (camera_params[f'{side}_root_xyz'].cuda().unsqueeze(1))

            if side == 'left':
                img_p = img_p * torch.Tensor([-1., 1., 1.]).cuda()

            if img_p.shape[1] != 0:
                sub_p = p[side] / 0.4
                batch_size, points_size = sub_p.shape[0], sub_p.shape[1]
                sub_p = sub_p.reshape(batch_size * points_size, -1)

            # finish swapping
            if pen:
                if side == 'left':
                    side = 'right'
                else:
                    side = 'left'

            batch_size = img_p.shape[0]

            if img_p.shape[1] != 0:
                img_p = torch.bmm(img_p, camera_params['R'].transpose(1,2).cuda()) + camera_params['T'].cuda().unsqueeze(1)
                root_z = img_p[:, 0, 2] 

                for i in range(batch_size):
                    img_p[i, :, 2] = img_p[i, :, 2] - root_z[i] 

                if side == 'left':
                    pt_feat = self.left_pt_embeddings(img_p.transpose(1,2))
                    pt_feat = torch.cat((img_p, pt_feat.transpose(1,2)), 2)
                    local_img_feat = self.img_ex_left(img_feat, pt_feat) 
                else:
                    pt_feat = self.right_pt_embeddings(img_p.transpose(1,2))
                    pt_feat = torch.cat((img_p, pt_feat.transpose(1,2)), 2)
                    local_img_feat = self.img_ex_right(img_feat, pt_feat) 

                local_img_feat = local_img_feat.reshape(local_img_feat.shape[0] * local_img_feat.shape[1], -1)

                p_feat = torch.cat((sub_p, local_img_feat), 1)

                local_c = c[side].repeat_interleave(points_size, dim=0)
                local_bone_lengths = bone_lengths[side].repeat_interleave(points_size, dim=0)

                if side == 'left':
                    left_batch_size, left_points_size = batch_size, points_size
                    left_p_r = self.left_decoder(p_feat.float(), local_c.float(), local_bone_lengths.float(), reduce_part=reduce_part)
                    left_p_r = self.left_decoder.sigmoid(left_p_r)
                else:
                    right_batch_size, right_points_size = batch_size, points_size
                    right_p_r = self.right_decoder(p_feat.float(), local_c.float(), local_bone_lengths.float(), reduce_part=reduce_part)
                    right_p_r = self.right_decoder.sigmoid(right_p_r)
            else:
                if side == 'left':
                    left_batch_size, left_points_size = 1, 0
                    left_p_r = torch.empty((1, 0)).cuda()
                else:
                    right_batch_size, right_points_size = 1, 0
                    right_p_r = torch.empty((1, 0)).cuda()

        if reduce_part:
            if right_points_size > 0:
                right_p_r, _ = right_p_r.max(1, keepdim=True)
            if left_points_size > 0:
                left_p_r, _ = left_p_r.max(1, keepdim=True)

            left_p_r = left_p_r.reshape(left_batch_size, left_points_size)
            right_p_r = right_p_r.reshape(right_batch_size, right_points_size)

        if test:
            final_left_p_r = torch.zeros((left_p.shape[1])).cuda()
            final_left_p_r[left_valid[0]] = left_p_r.squeeze()

            final_right_p_r = torch.zeros((right_p.shape[1])).cuda()
            final_right_p_r[right_valid[0]] = right_p_r.squeeze()

            return final_left_p_r.unsqueeze(0), final_right_p_r.unsqueeze(0)

        return left_p_r, right_p_r


    def to(self, device):
        ''' Puts the model to the device.
        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
