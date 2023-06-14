import os
import numpy as np
from tqdm import trange
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from artihand.utils import visualize as vis
from artihand.training import BaseTrainer
from artihand import diff_operators

# For dubugging
# from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# from mpl_toolkits.mplot3d import Axes3D
# from artihand.utils.visualize import visualize_pointcloud

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.
    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        skinning_loss_weight (float): skinning loss weight for part model
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples
    '''

    def __init__(self, model, optimizer, skinning_loss_weight=0, device=None, 
                 input_type='img', use_sdf=False, vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.skinning_loss_weight = skinning_loss_weight
        self.use_sdf = use_sdf
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.mse_loss = torch.nn.MSELoss()

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

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
        return loss_dict # loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.
        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Data
        # points = data.get('points').to(device)
        # occ = data.get('occ').to(device)

        img, mask, dense, camera_params, mano_data = data
        imgs = (img, mask, dense, camera_params)

        left_inputs = mano_data['left'].get('inputs').to(device)
        right_inputs = mano_data['right'].get('inputs').to(device)

        left_root_rot_mat = mano_data['left'].get('root_rot_mat').to(device)
        right_root_rot_mat = mano_data['right'].get('root_rot_mat').to(device)

        left_voxels_occ = mano_data['left'].get('voxels')
        right_voxels_occ = mano_data['right'].get('voxels')

        left_points_iou = mano_data['left'].get('points_iou.points').to(device)
        right_points_iou = mano_data['right'].get('points_iou.points').to(device)

        left_occ_iou = mano_data['left'].get('points_iou.occ').to(device)
        right_occ_iou = mano_data['right'].get('points_iou.occ').to(device)

        if self.model.use_bone_length:
            left_bone_lengths = mano_data['left'].get('bone_lengths').to(device)
            right_bone_lengths = mano_data['right'].get('bone_lengths').to(device)
        else:
            left_bone_lengths = right_bone_lengths = None

        kwargs = {}

        # with torch.no_grad():
        #     elbo, rec_error, kl = self.model.compute_elbo(
        #         points, occ, inputs, **kwargs)

        # eval_dict['loss'] = -elbo.mean().item()
        # eval_dict['rec_error'] = rec_error.mean().item()
        # eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = left_points_iou.size(0)

        with torch.no_grad():
            left_p_out, right_p_out, left_valid, right_valid = self.model(imgs, (left_root_rot_mat, right_root_rot_mat), (left_points_iou, right_points_iou), (left_inputs, right_inputs), bone_lengths=(left_bone_lengths, right_bone_lengths),
                               sample=self.eval_sample, **kwargs)

        left_occ_iou_np = (left_occ_iou >= 0.5).cpu().numpy()
        right_occ_iou_np = (right_occ_iou >= 0.5).cpu().numpy()

        if self.use_sdf:
            occ_iou_hat_np = (p_out <= threshold).cpu().numpy()
        else:
            left_occ_iou_hat_np = (left_p_out >= threshold).cpu().numpy()
            right_occ_iou_hat_np = (right_p_out >= threshold).cpu().numpy()


        sub_left_occ_iou_np = np.zeros(left_occ_iou_hat_np.shape)

        for batch_idx in range(sub_left_occ_iou_np.shape[0]):
            sub_left_occ_iou_np[batch_idx] = left_occ_iou_np[batch_idx, left_valid[batch_idx].cpu()]

        sub_right_occ_iou_np = np.zeros(right_occ_iou_hat_np.shape)

        for batch_idx in range(sub_right_occ_iou_np.shape[0]):
            sub_right_occ_iou_np[batch_idx] = right_occ_iou_np[batch_idx, right_valid[batch_idx].cpu()]

        print('left')
        print(left_p_out.cpu().numpy())
        print(sub_left_occ_iou_np)
        print('right')
        print(right_p_out.cpu().numpy())
        print(sub_right_occ_iou_np)
        print()


        left_iou = compute_iou(sub_left_occ_iou_np, left_occ_iou_hat_np).mean()
        right_iou = compute_iou(sub_right_occ_iou_np, right_occ_iou_hat_np).mean()
        eval_dict['iou'] = (left_iou + right_iou) / 2

        # import pdb; pdb.set_trace()
        # import numpy as np
        # sample_ids = np.random.choice(occ_iou_np.shape[1], 1024)
        # point_vis = points_iou[0, sample_ids].cpu().numpy()

        # point_in = points_iou[0, occ_iou_np[0, :]]
        # point_in_ids = np.random.choice(point_in.shape[0], 1024)
        # point_in_vis = point_in[point_in_ids].cpu().numpy()

        # point_pred_in = points_iou[0, occ_iou_hat_np[0, :]]
        # point_pred_in_ids = np.random.choice(point_pred_in.shape[0], 1024)
        # point_pred_in_vis = point_pred_in[point_pred_in_ids].cpu().numpy()

        # fig = plt.figure()
        # ax = fig.gca(projection=Axes3D.name)
        # ax.scatter(point_vis[:, 2], point_vis[:, 0], point_vis[:, 1], color='b')
        # ax.scatter(point_in_vis[:, 2], point_in_vis[:, 0], point_in_vis[:, 1], color='r')
        # # ax.scatter(surface_points[:, 2], surface_points[:, 0], surface_points[:, 1], color='orange')
        # plt.show()

        # Estimate voxel iou
        voxels_occ = None
        if voxels_occ is not None:
            import pdb; pdb.set_trace()
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.
        Args:
            data (dict): data dictionary
        '''
        device = self.device

        img, mask, dense, camera_params, mano_data = data
        imgs = (img, mask, dense, camera_params)

        batch_size = mano_data['left']['inputs'].size(0)

        left_inputs = mano_data['left'].get('inputs').to(device)
        right_inputs = mano_data['right'].get('inputs').to(device)
        inputs = (left_inputs, right_inputs)

        left_root_rot_mat = mano_data['left'].get('root_rot_mat').to(device)
        right_root_rot_mat = mano_data['right'].get('root_rot_mat').to(device)
        root_rot_mat = (left_root_rot_mat, right_root_rot_mat)

        if self.model.use_bone_length:
            left_bone_lengths = mano_data['left'].get('bone_lengths').to(device)
            right_bone_lengths = mano_data['right'].get('bone_lengths').to(device)
        else:
            left_bone_lengths = right_bone_lengths = None

        bone_lengths = (left_bone_lengths, right_bone_lengths)

        shape = (32, 32, 32)
        # shape = (64, 64, 64)
        left_p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        left_p = left_p.expand(batch_size, *left_p.size())

        right_p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        right_p = right_p.expand(batch_size, *right_p.size())

        p = (left_p, right_p)

        print(left_p.shape, right_p.shape)
        kwargs = {}
        with torch.no_grad():
            left_p_r, right_p_r, left_valid, right_valid = self.model(imgs, root_rot_mat, p, inputs, bone_lengths=bone_lengths, sample=self.eval_sample, **kwargs)

        print(left_p_r.shape, left_valid.shape)
        exit()
        left_occ_hat = left_p_r.view(batch_size, *shape)
        right_occ_hat = right_p_r.view(batch_size, *shape)

        if self.use_sdf:
            voxels_out = (occ_hat <= self.threshold).cpu().numpy()
            print('Not Implemented Error')
            exit()
        else:
            left_voxels_out = (left_occ_hat >= self.threshold).cpu().numpy()
            right_voxels_out = (right_occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            # vis.visualize_data(
            #     inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                left_voxels_out[i], os.path.join(self.vis_dir, 'left_%03d.png' % i))
            vis.visualize_voxels(
                right_voxels_out[i], os.path.join(self.vis_dir, 'right_%03d.png' % i))


    def compute_skinning_loss(self, c, data, bone_lengths=None):
        ''' Computes skinning loss for part-base regularization.
        Args:
            c (tensor): encoded latent vector
            data (dict): data dictionary
            bone_lengths (tensor): bone lengths
        '''
        device = self.device
        p = data.get('mesh_verts').to(device)
        labels = data.get('mesh_vert_labels').to(device)
        
        batch_size, points_size, p_dim = p.size()
        kwargs = {}

        pred = self.model.decode(p, c, bone_lengths=bone_lengths, reduce_part=False, **kwargs)

        labels = labels.long()
        # print("label", labels.size(), labels.type())

        if self.use_sdf:
            level_set = 0.0
        else:
            level_set = 0.5

        labels = F.one_hot(labels, num_classes=pred.size(-1)).float()
        labels = labels * level_set
        # print("label", labels.size(), labels.type())
        # print("label size", *labels.size())
        pred = pred.view(batch_size, points_size, pred.size(-1))
        # print("pred", pred.size())
        # print("occ", occ)
        sk_loss = self.mse_loss(pred, labels)
        
        return sk_loss
    
    def compute_sdf_loss(self, pred_sdf, input_points, surface_normals, surface_point_size):
        '''
            x: batch of input coordinates
            y: usually the output of the trial_soln function
        '''
        # gt_sdf = gt['sdf']
        # gt_normals = gt['normals']

        # coords = model_output['model_in']
        # pred_sdf = model_output['model_out']

        # gradient = diff_operators.gradient(pred_sdf, coords)

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        # Surface points on the left, off-surface on the right
        # sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
        # inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
        # normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
        #                                 torch.zeros_like(gradient[..., :1]))
        # grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

        gradient = diff_operators.gradient(pred_sdf, input_points)
        surface_gradient = gradient[:, :surface_point_size]

        surface_pred = pred_sdf[:, :surface_point_size]
        off_pred = pred_sdf[:, surface_point_size:]

        # import pdb; pdb.set_trace()

        sdf_constraint = surface_pred 
        inter_constraint = torch.exp(-1e2 * torch.abs(off_pred))
        normal_constraint = 1 - F.cosine_similarity(surface_gradient, surface_normals, dim=-1)[..., None]
        # normal_constraint = torch.zeros_like(gradient[..., :1])
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
        
        # Exp      # Lapl
        # -----------------
        return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
                'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
                'normal_constraint': normal_constraint.mean() * 1e2, # 1e2,  # 1e2
                'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1
        # inter = 3e3 for ReLU-PE
        
    def compute_loss(self, data):
        ''' Computes the loss.
        Args:
            data (dict): data dictionary
        '''
        device = self.device

        # Point input
        if self.use_sdf:
            points = data.get('points').to(device)
            normals = data.get('normals').to(device)
            off_points = data.get('off_points').to(device)

            points.requires_grad_(True)
            normals.requires_grad_(True)
            off_points.requires_grad_(True)

            # import pdb; pdb.set_trace()
            surface_point_size = points.shape[1]
            input_points = torch.cat([points, off_points], 1)

            # points_np = points[0, :512].detach().cpu().numpy()
            # normals_np = normals[0, :512].detach().cpu().numpy()
            # off_points_np = off_points[0, :512].detach().cpu().numpy()

            # surface_points = data.get('mesh_verts')[0, :512].detach().cpu().numpy()
            # import pdb; pdb.set_trace()
            # fig = plt.figure()
            # ax = fig.gca(projection=Axes3D.name)
            # ax.scatter(points_np[:, 2], points_np[:, 0], points_np[:, 1], color='b')
            # # ax.scatter(off_points_np[:, 2], off_points_np[:, 0], off_points_np[:, 1], color='r')
            # ax.scatter(surface_points[:, 2], surface_points[:, 0], surface_points[:, 1], color='orange')
            
            
            # mesh_verts
            # visualize_pointcloud(points_np, normals_np, off_points_np, show=True)
            # visualize_pointcloud(points_np, None, off_points_np, show=True)
            

        else:
            img, mask, dp, camera_params, mano_data = data

            left_p = mano_data['left'].get('points').to(device)
            left_occ = mano_data['left'].get('occ').to(device)
            left_root_rot_mat = mano_data['left'].get('root_rot_mat').to(device)
            left_inputs = mano_data['left'].get('inputs').to(device)

            right_p = mano_data['right'].get('points').to(device)
            right_occ = mano_data['right'].get('occ').to(device)
            right_root_rot_mat = mano_data['right'].get('root_rot_mat').to(device)
            right_inputs = mano_data['right'].get('inputs').to(device)

        hms, mask_pred, dp_pred, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img) 
        #print(mask.shape, dp.shape)
        #print(hms.shape, mask_pred.shape, dp_pred.shape)
        #exit()

        img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1] 
        #img_f, dp_f = img_fmaps[-1], dp_fmaps[-1] 


        img_f = nn.functional.interpolate(img_f, size=[256, 256], mode='bilinear')
        hms_f = nn.functional.interpolate(hms_f, size=[256, 256], mode='bilinear') 
        dp_f = nn.functional.interpolate(dp_f, size=[256, 256], mode='bilinear') 

        #img_feat = torch.cat((img_f, hms_f, dp_f), 1)
        img_feat = torch.cat((hms_f, dp_f), 1)
        img_feat = self.model.image_final_layer(img_feat)
        #img_feat = img_f

        kwargs = {}
        if self.model.use_bone_length:
            left_bone_lengths = mano_data['left'].get('bone_lengths').to(device)
            right_bone_lengths = mano_data['right'].get('bone_lengths').to(device)
        else:
            left_bone_lengths = right_bone_lengths = None
        
        loss_dict = {}

        # comment it out as we do not use an encoder for now
        #c = self.model.encode_inputs(inputs)
        left_c = left_inputs
        right_c = right_inputs

        if self.use_sdf:
            pred = self.model.decode(input_points, c, bone_lengths=bone_lengths, **kwargs)
            losses = self.compute_sdf_loss(pred, input_points, normals, surface_point_size)
            loss = 0.
            # (sdf, inter, normal_constraint, grad_constraint)
            for loss_name, loss_value in losses.items():
                loss += loss_value.mean() 
                loss_dict[loss_name] = loss_value.item()

        else:
            left_pred, right_pred, left_valid, right_valid = self.model.decode(img_feat, camera_params, (left_root_rot_mat, right_root_rot_mat), (left_p, right_p), (left_c, right_c), bone_lengths=(left_bone_lengths, right_bone_lengths), **kwargs)

            # print("pred", pred)
            # print("occ", occ)

            sub_left_occ = torch.zeros(left_valid.shape).to(device) 
            sub_right_occ = torch.zeros(right_valid.shape).to(device) 

            for batch_idx in range(left_occ.shape[0]):
                sub_left_occ[batch_idx] = left_occ[batch_idx, left_valid[batch_idx]]

            for batch_idx in range(right_occ.shape[0]):
                sub_right_occ[batch_idx] = right_occ[batch_idx, right_valid[batch_idx]]
            
            '''
            print('left')
            print(left_pred)
            print(sub_left_occ)
            print()
            print('right')
            print(right_pred)
            print(sub_right_occ)
            print()
            '''

            loss = self.mse_loss(left_pred, sub_left_occ) + self.mse_loss(right_pred, sub_right_occ)

            loss_dict['occ'] = loss.item()

        if self.skinning_loss_weight > 0:
            sk_loss = self.compute_skinning_loss(c, mano_data, bone_lengths=bone_lengths)
            loss_dict['skin'] = sk_loss.item()
            loss = loss + self.skinning_loss_weight * sk_loss
        
        # import pdb; pdb.set_trace()
        loss_dict['total'] = loss.item()
        return loss, loss_dict
