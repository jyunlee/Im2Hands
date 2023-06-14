import os
import sys
import torch

import torch.nn as nn
import numpy as np

from torch.nn import functional as F
from torch import distributions as dist
from tqdm import trange

from im2mesh.common import (
    compute_iou, make_3d_grid
)
from artihand.utils import visualize as vis
from artihand.training import BaseTrainer
from artihand import diff_operators
from dependencies.halo.halo_adapter.transform_utils import xyz_to_xyz1   


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.'''

    def __init__(self, model, optimizer, skinning_loss_weight=0, device=None, 
                 input_type='img', threshold=0.5, eval_sample=False):

        self.model = model
        self.optimizer = optimizer
        self.skinning_loss_weight = skinning_loss_weight
        self.device = device
        self.input_type = input_type
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.mse_loss = torch.nn.MSELoss()


    def train_step(self, data):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
        '''

        self.model.train()
        self.optimizer.zero_grad()

        loss, loss_dict = self.compute_loss(data)
        loss.backward()

        self.optimizer.step()

        return loss_dict


    def eval_step(self, data):
        ''' Performs an evaluation step.
        Args:
            data (dict): data dictionary
        '''

        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        img, camera_params, mano_data, _ = data

        inputs = {'left': mano_data['left'].get('inputs').to(device),
                  'right': mano_data['right'].get('inputs').to(device)}

        points = {'left': mano_data['left'].get('points_iou.points').to(device),
                  'right': mano_data['right'].get('points_iou.points').to(device)}

        root_rot_mat = {'left': mano_data['left'].get('root_rot_mat').to(device),
                        'right': mano_data['right'].get('root_rot_mat').to(device)}

        bone_lengths = {'left': mano_data['left'].get('bone_lengths').to(device),
                        'right': mano_data['right'].get('bone_lengths').to(device)}

        occ_iou = {'left': mano_data['left'].get('points_iou.occ').to(device),
                   'right': mano_data['right'].get('points_iou.occ').to(device)}

        kwargs = {}

        with torch.no_grad():
            left_occ, right_occ = self.model(img, camera_params, inputs, points, root_rot_mat, bone_lengths, sample=self.eval_sample, **kwargs)

        # evaluation 
        left_occ_iou_np = (occ_iou['left'] >= 0.5).cpu().numpy()
        right_occ_iou_np = (occ_iou['right'] >= 0.5).cpu().numpy()

        left_occ_iou_hat_np = (left_occ >= threshold).cpu().numpy()
        right_occ_iou_hat_np = (right_occ >= threshold).cpu().numpy()

        left_iou = compute_iou(left_occ_iou_np, left_occ_iou_hat_np).mean()
        right_iou = compute_iou(right_occ_iou_np, right_occ_iou_hat_np).mean()

        eval_dict['iou'] = (left_iou + right_iou) / 2

        batch_size = points['left'].size(0)

        return eval_dict


    def compute_skinning_loss(self, c, data, bone_lengths=None):
        ''' Computes skinning loss for part-base regularization.'''

        device = self.device
        p = data.get('mesh_verts').to(device)
        labels = data.get('mesh_vert_labels').to(device)
        
        batch_size, points_size, p_dim = p.size()
        kwargs = {}

        pred = self.model.decode(p, c, bone_lengths=bone_lengths, reduce_part=False, **kwargs)

        labels = labels.long()
        level_set = 0.5

        labels = F.one_hot(labels, num_classes=pred.size(-1)).float()
        labels = labels * level_set
        pred = pred.view(batch_size, points_size, pred.size(-1))
        sk_loss = self.mse_loss(pred, labels)
        
        return sk_loss
    

    def compute_loss(self, data):
        ''' Computes the loss.
        Args:
            data (dict): data dictionary
        '''
        device = self.device
        self.model = self.model.to(device)

        threshold = self.threshold

        img, camera_params, mano_data, _ = data

        inputs = {'left': mano_data['left'].get('inputs').to(device),
                  'right': mano_data['right'].get('inputs').to(device)}

        points = {'left': mano_data['left'].get('points').to(device),
                  'right': mano_data['right'].get('points').to(device)}

        root_rot_mat = {'left': mano_data['left'].get('root_rot_mat').to(device),
                        'right': mano_data['right'].get('root_rot_mat').to(device)}

        bone_lengths = {'left': mano_data['left'].get('bone_lengths').to(device),
                        'right': mano_data['right'].get('bone_lengths').to(device)}

        occ = {'left': mano_data['left'].get('occ').to(device),
               'right': mano_data['right'].get('occ').to(device)}

        kwargs = {}

        left_occ, right_occ = self.model(img, camera_params, inputs, points, root_rot_mat, bone_lengths, sample=self.eval_sample, **kwargs)

        loss_dict = {}

        occ_loss = self.mse_loss(left_occ, occ['left']) + self.mse_loss(right_occ, occ['right'])

        loss_dict['occ'] = occ_loss.item()

        if self.skinning_loss_weight > 0:
            sk_loss = self.compute_skinning_loss(c, mano_data, bone_lengths=bone_lengths)
            loss_dict['skin'] = sk_loss.item()
            loss = loss + self.skinning_loss_weight * sk_loss

        loss = occ_loss

        loss_dict['total'] = loss.item()

        return loss, loss_dict

