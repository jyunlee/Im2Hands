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


def preprocess_joints(left_joints, right_joints, camera_params, root_rot_mat):

    # preprocess left joints
    #left_joints = torch.bmm(xyz_to_xyz1(left_joints.double()), root_rot_mat['left'].double())[:, :, :3]
    left_joints = left_joints + (camera_params['left_root_xyz'].cuda().unsqueeze(1))
    left_joints = left_joints * torch.Tensor([-1., 1., 1.]).cuda()

    left_joints = torch.bmm(left_joints, camera_params['R'].transpose(1,2).double().cuda()) + camera_params['T'].double().cuda().unsqueeze(1)

    # preprocess right joints
    #right_joints = torch.bmm(xyz_to_xyz1(right_joints.double()), root_rot_mat['right'].double())[:, :, :3]
    right_joints = right_joints + (camera_params['right_root_xyz'].cuda().unsqueeze(1))

    right_joints = torch.bmm(right_joints, camera_params['R'].transpose(1,2).double().cuda()) + camera_params['T'].double().cuda().unsqueeze(1)

    # normalize
    left_mid_joint = left_joints[:, 4, :].unsqueeze(1)

    left_joints = left_joints - left_mid_joint
    right_joints = right_joints - left_mid_joint

    return left_joints*1000, right_joints*1000


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.
    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        skinning_loss_weight (float): skinning loss weight for part model
        device (device): pytorch device
        input_type (str): input type
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples
    '''

    def __init__(self, model, optimizer, skinning_loss_weight=0, device=None, 
                 input_type='img', threshold=0.5, eval_sample=False):

        self.model = model
        self.optimizer = optimizer
        self.skinning_loss_weight = skinning_loss_weight
        self.device = device
        self.input_type = input_type
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.loss = torch.nn.MSELoss()


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

        joints_gt = {'left': mano_data['left'].get('joints').to(device),
                     'right': mano_data['right'].get('joints').to(device)}

        joints = {'left': mano_data['left'].get('pred_joints').to(device),
                  'right': mano_data['right'].get('pred_joints').to(device)}

        root_rot_mat = {'left': mano_data['left'].get('root_rot_mat').to(device),
                        'right': mano_data['left'].get('root_rot_mat').to(device)}

        kwargs = {}

        # joint space conversion & normalization
        left_joints, right_joints = preprocess_joints(joints['left'], joints['right'], camera_params, root_rot_mat)
        left_joints_gt, right_joints_gt = preprocess_joints(joints_gt['left'], joints_gt['right'], camera_params, root_rot_mat)

        in_joints = {'left': left_joints, 'right': right_joints}

        with torch.no_grad():
            left_joints_pred, right_joints_pred = self.model(img, camera_params, in_joints, **kwargs) 

        left_joints_pred = left_joints_pred - left_joints_pred[:, 4, :].unsqueeze(1) 
        right_joints_pred = right_joints_pred - right_joints_pred[:, 4, :].unsqueeze(1) 

        left_joints_gt = left_joints_gt - left_joints_gt[:, 4, :].unsqueeze(1) 
        right_joints_gt = right_joints_gt - right_joints_gt[:, 4, :].unsqueeze(1) 

        left_joints = left_joints - left_joints[:, 4, :].unsqueeze(1) 
        right_joints = right_joints - right_joints[:, 4, :].unsqueeze(1) 

        eval_dict['joint_err'] = self.loss(left_joints_pred.to(torch.float64), left_joints_gt).item() \
                                 + self.loss(right_joints_pred.to(torch.float64), right_joints_gt).item() 

        joints_num = left_joints_pred.shape[0]
        left_jpe, right_jpe = 0., 0.
        in_left_jpe, in_right_jpe = 0., 0.

        '''
        import open3d as o3d
        import cv2
        img = cv2.imread(camera_params['img_path'][0])
        cv2.imwrite('debug/check.jpg', img)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(left_joints_pred[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/left_joints_pred.ply", pcd)
        pcd.points = o3d.utility.Vector3dVector(right_joints_pred[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/right_joints_pred.ply", pcd)
        pcd.points = o3d.utility.Vector3dVector(left_joints_gt[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/left_joints_gt.ply", pcd)
        pcd.points = o3d.utility.Vector3dVector(right_joints_gt[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/right_joints_gt.ply", pcd)
        pcd.points = o3d.utility.Vector3dVector(left_joints[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/left_joints.ply", pcd)
        pcd.points = o3d.utility.Vector3dVector(right_joints[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/right_joints.ply", pcd)

        exit()
        '''
        for i in range(joints_num):
            left_jpe += torch.linalg.norm((left_joints_pred[i] - left_joints_gt[i]), ord=2, dim=-1).mean().item()
            right_jpe += torch.linalg.norm((right_joints_pred[i] - right_joints_gt[i]), ord=2, dim=-1).mean().item() 
            in_left_jpe += torch.linalg.norm((left_joints[i] - left_joints_gt[i]), ord=2, dim=-1).mean().item() 
            in_right_jpe += torch.linalg.norm((right_joints[i] - right_joints_gt[i]), ord=2, dim=-1).mean().item() 

        left_jpe /= joints_num
        right_jpe /= joints_num
        in_left_jpe /= joints_num
        in_right_jpe /= joints_num

        eval_dict['jpe'] = (left_jpe + right_jpe) * 0.5
        eval_dict['in_jpe'] = (in_left_jpe + in_right_jpe) * 0.5

        return eval_dict
   

    def compute_loss(self, data):
        ''' Computes the loss.
        Args:
            data (dict): data dictionary
        '''
        device = self.device
        self.model = self.model.to(device)

        threshold = self.threshold

        img, camera_params, mano_data, _ = data

        joints_gt = {'left': mano_data['left'].get('joints').to(device),
                     'right': mano_data['right'].get('joints').to(device)}

        joints = {'left': mano_data['left'].get('pred_joints').to(device),
                  'right': mano_data['right'].get('pred_joints').to(device)}

        root_rot_mat = {'left': mano_data['left'].get('root_rot_mat').to(device),
                        'right': mano_data['left'].get('root_rot_mat').to(device)}

        kwargs = {}

        # joint space conversion & normalization
        left_joints, right_joints = preprocess_joints(joints['left'], joints['right'], camera_params, root_rot_mat)
        left_joints_gt, right_joints_gt = preprocess_joints(joints_gt['left'], joints_gt['right'], camera_params, root_rot_mat)

        in_joints = {'left': left_joints, 'right': right_joints}

        left_joints_pred, right_joints_pred = self.model(img, camera_params, in_joints, **kwargs) 

        left_joints_pred = left_joints_pred - left_joints_pred[:, 4, :].unsqueeze(1) 
        right_joints_pred = right_joints_pred - right_joints_pred[:, 4, :].unsqueeze(1) 

        left_joints_gt = left_joints_gt - left_joints_gt[:, 4, :].unsqueeze(1) 
        right_joints_gt = right_joints_gt - right_joints_gt[:, 4, :].unsqueeze(1) 

        loss = self.loss(left_joints_pred.to(torch.float64), left_joints_gt) \
               + self.loss(right_joints_pred.to(torch.float64), right_joints_gt)

        loss_dict = {}
        loss_dict['total'] = loss.item()

        return loss, loss_dict

