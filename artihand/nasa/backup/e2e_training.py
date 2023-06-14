import os
from tqdm import trange
import torch
from torch.nn import functional as F
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

def list2dev(L, dev):
    assert isinstance(L, list)
    return [ten.to(dev) for ten in L]

def dict2dev(d, dev, selected_keys=None):
    """
    If selected_keys is not None, then only move to the device for those keys
    """
    types = (torch.Tensor, torch.nn.utils.rnn.PackedSequence)
    if selected_keys is None:
        selected_keys = list(d.keys())
    for k, v in d.items():
        if k in selected_keys:
            if isinstance(v, types):
                d[k] = d[k].to(dev)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], types):
                d[k] = list2dev(v, dev)
    return d


class E2ETrainer(BaseTrainer):
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
        self.model = model.to(device)
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

        #inputs = data.get('inputs').to(device)
        #voxels_occ = data.get('voxels')

        inputs, targets, meta_info, left_data, right_data = data

        batch_size = inputs['img'].size(0)
        inputs = dict2dev(inputs, device)
        targets = dict2dev(targets, device)
        meta_info = dict2dev(meta_info, device)

        left_p = left_data.get('points').to(device)
        right_p = right_data.get('points').to(device)

        left_points_iou = left_data.get('points_iou.points').to(device)
        left_occ_iou = left_data.get('points_iou.occ').to(device)
        right_points_iou = right_data.get('points_iou.points').to(device)
        right_occ_iou = right_data.get('points_iou.occ').to(device)

        '''
        if self.model.use_bone_length:
            bone_lengths = data.get('bone_lengths').to(device)
        else:
            bone_lengths = None
        '''
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

            #p_out = self.model(points_iou, inputs, bone_lengths=bone_lengths,
            #                   sample=self.eval_sample, **kwargs)
            left_p_out, right_p_out, _ = self.model(inputs, targets, meta_info, left_p, right_p)

        left_occ_iou_np = (left_occ_iou >= 0.5).cpu().numpy()
        right_occ_iou_np = (right_occ_iou >= 0.5).cpu().numpy()
        if self.use_sdf:
            occ_iou_hat_np = (p_out <= threshold).cpu().numpy()
        else:
            left_occ_iou_hat_np = (left_p_out >= threshold).cpu().numpy()
            right_occ_iou_hat_np = (right_p_out >= threshold).cpu().numpy()
        
        
        '''
        torch.save(left_occ_iou_hat_np, 'left_occ_iou_hat_np.pt')
        torch.save(left_occ_iou_np, 'left_occ_iou_np.pt')
        torch.save(right_occ_iou_hat_np, 'right_occ_iou_hat_np.pt')
        torch.save(right_occ_iou_np, 'right_occ_iou_np.pt')
        print('occ saved!')
        exit()
        '''
        left_iou = compute_iou(left_occ_iou_np, left_occ_iou_hat_np).mean()
        right_iou = compute_iou(right_occ_iou_np, right_occ_iou_hat_np).mean()
        eval_dict['left_iou'] = left_iou
        eval_dict['right_iou'] = right_iou
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

        inputs, targets, meta_info, left_data, right_data = data

        batch_size = inputs['img'].size(0)
        inputs = dict2dev(inputs, device)
        targets = dict2dev(targets, device)
        meta_info = dict2dev(meta_info, device)

        left_p = left_data.get('points').to(device)
        right_p = right_data.get('points').to(device)

        '''
        if self.model.use_bone_length:
            bone_lengths = data.get('bone_lengths').to(device)
        else:
            bone_lengths = None
        '''
        shape = (32, 32, 32)
        # shape = (64, 64, 64)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            left_p_r, right_p_r, _ = self.model(inputs, targets, meta_info, p, p)

        left_occ_hat = left_p_r.view(batch_size, *shape)
        right_occ_hat = right_p_r.view(batch_size, *shape)

        if self.use_sdf:
            voxels_out = (occ_hat <= self.threshold).cpu().numpy()
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

        # modify here!!!
        else:
            inputs, targets, meta_info, left_data, right_data = data

            inputs = dict2dev(inputs, device)
            targets = dict2dev(targets, device)
            meta_info = dict2dev(meta_info, device)

            left_p = left_data.get('points').to(device)
            left_occ = left_data.get('occ').to(device)

            right_p = right_data.get('points').to(device)
            right_occ = right_data.get('occ').to(device)


        #inputs = data.get('inputs').to(device)

        #kwargs = {}
        #if self.model.use_bone_length:
        #    bone_lengths = data.get('bone_lengths').to(device)
        #else:
        #    bone_lengths = None
        
        loss_dict = {}

        #c = self.model.encode_inputs(inputs)

        if self.use_sdf:
            pred = self.model.decode(input_points, c, bone_lengths=bone_lengths, **kwargs)
            losses = self.compute_sdf_loss(pred, input_points, normals, surface_point_size)
            loss = 0.
            # (sdf, inter, normal_constraint, grad_constraint)
            for loss_name, loss_value in losses.items():
                loss += loss_value.mean() 
                loss_dict[loss_name] = loss_value.item()

        # modify here!!!
        else:
            left_pred, right_pred, digit_loss = self.model(inputs, targets, meta_info, left_p, right_p)

            '''            
            import open3d as o3d
            vis_kps = left_p[0].detach().cpu().numpy() * 0.4
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_kps)
            o3d.io.write_point_cloud("query_points.ply", pcd)
            '''

            #pred = self.model.decode(p, c, bone_lengths=bone_lengths, **kwargs)
            digit_loss = {k: digit_loss[k].mean().view(-1) for k in digit_loss if k != 'kp_cam'}
            digit_total_loss = sum(digit_loss[k] for k in digit_loss)
            loss_dict['digit'] = digit_total_loss 

            loss = self.mse_loss(left_pred, left_occ) + self.mse_loss(right_pred, right_occ) 
            loss_dict['occ'] = loss.item()

            total_loss = loss + digit_total_loss
            
        if self.skinning_loss_weight > 0:
            sk_loss = self.compute_skinning_loss(c, data, bone_lengths=bone_lengths)
            loss_dict['skin'] = sk_loss.item()
            loss = loss + self.skinning_loss_weight * sk_loss
        
        # import pdb; pdb.set_trace()
        loss_dict['total'] = loss.item() + digit_total_loss.item()
        return total_loss, loss_dict
