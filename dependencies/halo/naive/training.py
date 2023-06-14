import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
# from im2mesh.common import (
#     compute_iou, make_3d_grid
# )
from models.utils import visualize as vis
from models.training import BaseTrainer

from models.naive.loss.loss import (BoneLengthLoss, RootBoneAngleLoss, AllBoneAngleLoss,
                                    SurfaceDistanceLoss, InterpenetrationLoss, ManoVertLoss)

# mano loss
import sys
sys.path.insert(0, "/home/korrawe/halo_vae/scripts")
from manopth.manolayer import ManoLayer
from manopth import demo

# temp
import trimesh
from trimesh.base import Trimesh

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    def __init__(self, model, optimizer, kl_weight=0.1, device=None,
                 input_type='img', vis_dir=None, threshold=0.5, eval_sample=False, 
                 use_inter_loss=False, use_refine_net=False, use_mano_loss=False):
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.bone_length_loss = BoneLengthLoss(device=device)
        self.root_bone_angle_loss = RootBoneAngleLoss(device=device)
        self.all_bone_angle_loss = AllBoneAngleLoss(device=device)
        self.surface_dist_loss = SurfaceDistanceLoss(device=device)
        self.inter_loss = InterpenetrationLoss(device=device)

        self.use_surface_loss = False  # True
        self.use_inter_loss = use_inter_loss
        self.use_refine_net = use_refine_net

        self.use_refine_net = False

        self.use_mano_loss = use_mano_loss
        if use_mano_loss:
            self.mano_loss = ManoVertLoss(device=device)
            self.mano_layer = ManoLayer(
                mano_root='/home/korrawe/halo_vae/scripts/mano/models', center_idx=0, use_pca=True, ncomps=45, flat_hand_mean=False)
            self.mano_layer = self.mano_layer.to(device)

        # mano_joint_parent
        self.joint_parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
        # self.root_bone_idx = np.array([1, 5, 9, 13, 17])

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, epoch_it):
        ''' Performs a training step.
        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_dict = self.compute_loss(data, epoch_it)
        loss.backward()
        self.optimizer.step()
        return loss_dict  # loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.
        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        # points = data.get('points').to(device)
        # occ = data.get('occ').to(device)
        object_points = data.get('object_points').float().to(device)
        hand_joints = data.get('hand_joints').float().to(device)
        # If use BPS
        if self.model.use_bps:
            object_points = data.get('object_bps').float().to(device)

        # inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        # voxels_occ = data.get('voxels')

        # points_iou = data.get('points_iou.points').to(device)
        # occ_iou = data.get('points_iou.occ').to(device)

        # if self.model.use_bone_length:
        #     bone_lengths = data.get('bone_lengths').to(device)
        # else:
        #     bone_lengths = None

        kwargs = {}

        with torch.no_grad():
            kl, pred, obj_c = self.model.compute_kl_divergence(
                object_points, hand_joints, reture_obj_latent=True
            )
            # elbo, rec_error, kl = self.model.compute_elbo()

        if self.use_mano_loss:
            hand_verts = data.get('hand_verts').float().to(device)
            rot, pose, shape, trans = pred[:, :3], pred[:, 3:48], pred[:, 48:58], pred[:, 58:61]
            eval_dict['vert'] = self.mano_loss(rot, pose, shape, trans, hand_verts).item()
            eval_dict['kl'] = kl.mean().item()
            eval_dict['loss'] = self.kl_weight * eval_dict['kl'] + eval_dict['vert']
            return eval_dict

        eval_dict['joints_recon'] = self.mse_loss(pred, hand_joints).item()
        eval_dict['bone_length'] = self.bone_length_loss(pred, hand_joints).item()
        root_bone_angle, root_plane_angle = self.root_bone_angle_loss(pred, hand_joints)
        eval_dict['root_bone_angle'] = root_bone_angle.item()
        eval_dict['root_plane_angle'] = root_plane_angle.item()
        eval_dict['all_bone_angle'] = self.all_bone_angle_loss(pred, hand_joints).item()
        # import pdb; pdb.set_trace()
        eval_dict['kl'] = kl.mean().item()

        eval_dict['loss'] = (
            2.0 * eval_dict['joints_recon']
            + 2.0 * eval_dict['bone_length']
            + 1.5 * eval_dict['root_bone_angle']
            + 1.5 * eval_dict['root_plane_angle']
            + 1.0 * eval_dict['all_bone_angle']
            + self.kl_weight * eval_dict['kl']
        )

        # Distance to object surface loss
        if self.use_surface_loss:
            gt_surface_dist = data.get('closest_point_dist').float().to(device)
            loss_surface_dist = self.surface_dist_loss(pred, object_points, gt_surface_dist)
            eval_dict['surface_dist'] = loss_surface_dist.item()
            # eval_dict['loss'] += 0.5 * eval_dict['surface_dist']

        # RefineNet loss
        if self.use_refine_net:
            tip_dists = data.get('tip_dists').float().to(device)
            noisy_joints = data.get('noisy_joints').float().to(device)
            refined_joints = self.model.refine_net(noisy_joints, obj_c, tip_dists)
            loss_refinement = self.mse_loss(refined_joints, hand_joints)
            eval_dict['refine_recon'] = loss_refinement.item()
            eval_dict['loss'] += 2.0 * eval_dict['refine_recon']

            loss_bone_length_re = self.bone_length_loss(refined_joints, hand_joints)
            loss_root_bone_angle_re, loss_root_plane_angle_re = self.root_bone_angle_loss(refined_joints, hand_joints)
            loss_all_bone_angle_re = self.all_bone_angle_loss(refined_joints, hand_joints)

            eval_dict['refine_other'] = (
                2.0 * loss_bone_length_re.item()
                + 1.5 * loss_root_bone_angle_re.item()
                + 1.5 * loss_root_plane_angle_re.item()
                + 1.0 * loss_all_bone_angle_re.item()
            )
            eval_dict['loss'] += eval_dict['refine_other']

        # Interpenetration loss
        if self.use_inter_loss:
            inside_points = data.get('inside_points').float().to(device)
            loss_inter, _ = self.inter_loss(pred, inside_points, self.model.halo_adapter)
            eval_dict['inter'] = loss_inter.item()
            eval_dict['loss'] += 4.0 * eval_dict['inter']  # 5.0

        # eval_dict['loss'] = -elbo.mean().item()
        # eval_dict['rec_error'] = rec_error.mean().item()

        # # Compute iou
        # batch_size = points.size(0)

        # with torch.no_grad():
        #     p_out = self.model(points_iou, inputs, bone_lengths=bone_lengths,
        #                        sample=self.eval_sample, **kwargs)

        # occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        # occ_iou_hat_np = (p_out >= threshold).cpu().numpy()
        # iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        # eval_dict['iou'] = iou

        # Estimate voxel iou
        # if voxels_occ is not None:
        #     voxels_occ = voxels_occ.to(device)
        #     points_voxels = make_3d_grid(
        #         (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
        #     points_voxels = points_voxels.expand(
        #         batch_size, *points_voxels.size())
        #     points_voxels = points_voxels.to(device)
        #     with torch.no_grad():
        #         p_out = self.model(points_voxels, inputs,
        #                            sample=self.eval_sample, **kwargs)

        #     voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
        #     occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
        #     iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

        #     eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data, epoch):
        ''' Performs a visualization step for the data.
        Args:
            data (dict): data dictionary
            epoch (int): epoch number
        '''
        device = self.device

        object_points = data.get('object_points').float().to(device)
        hand_joints_gt = data.get('hand_joints').float().to(device)
        object_inputs = object_points
        # If use BPS
        if self.model.use_bps:
            object_inputs = data.get('object_bps').float().to(device)

        vis_idx = np.random.randint(64)
        object_inputs = object_inputs[vis_idx].unsqueeze(0)
        object_points = object_points[vis_idx].unsqueeze(0)
        hand_joints_gt = hand_joints_gt[vis_idx].unsqueeze(0)

        # import pdb; pdb.set_trace()
        if self.use_mano_loss:
            hand_verts_gt = data.get('hand_verts').float().to(device)

        num_sample = 8
        for n in range(num_sample):
            # output_joints = self.model(object_points, hand_joints=hand_joints_gt, sample=False)  # sample=True
            if n == 0:
                output_joints = self.model(object_inputs, hand_joints=hand_joints_gt, sample=False)
            elif n == 1:
                output_joints = self.model(object_inputs, sample=False)
            else:
                output_joints = self.model(object_inputs, sample=True)
            # print("------------------")
            # print(output_joints)
            if self.use_mano_loss:
                rot, pose, shape, trans = output_joints[:, :3], output_joints[:, 3:48], output_joints[:, 48:58], output_joints[:, 58:61]
                _, output_joints = self.mano_layer(torch.cat((rot, pose), 1), shape, trans)
                output_joints = output_joints / 10.0

            print("val error: ", self.mse_loss(output_joints, hand_joints_gt))

            object_points_vis = object_points.detach().cpu().numpy()[0]  # [vis_idx]
            output_joints_vis = output_joints.detach().cpu().numpy()[0]  # [vis_idx]
            gt_joints = hand_joints_gt.detach().cpu().numpy()[0]  # [vis_idx]

            output_path = os.path.join(self.vis_dir, 'ep%03d_%03d_%03d.png' % (epoch, vis_idx, n))
            col = 'b' if n == 0 else 'g'
            vis.visualise_skeleton(output_joints_vis, object_points_vis, joint_order='mano', color=col, out_file=output_path, show=False)

            # HALO
            if self.model.halo_adapter is not None:
                output_mesh = self.model.halo_adapter(output_joints, joint_order='mano', return_kps=False, original_position=True)
                meshout_path = os.path.join(self.vis_dir, 'ep%03d_%03d_%03d.obj' % (epoch, vis_idx, n))
                if output_mesh is not None:
                    output_mesh.export(meshout_path)

                # Ground truth object
                gt_object_points = Trimesh(vertices=object_points_vis)
                objout_path = os.path.join(self.vis_dir, 'ep%03d_%03d_obj.obj' % (epoch, vis_idx))
                gt_object_points.export(objout_path)

                # import pdb; pdb.set_trace()
                # output_joints_new = self.model.halo_adapter(output_joints)
                # output_joints_new_vis = output_joints_new.detach().cpu().numpy()[0]  # [vis_idx]
                # vis.visualise_skeleton(output_joints_new_vis, object_points_vis, color=col, out_file=output_path, show=True)

        # mano_joint_parent = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

        # batch_size = data['points'].size(0)
        # inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        # if self.model.use_bone_length:
        #     bone_lengths = data.get('bone_lengths').to(device)
        # else:
        #     bone_lengths = None

        # shape = (32, 32, 32)
        # # shape = (64, 64, 64)
        # p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        # p = p.expand(batch_size, *p.size())

        # kwargs = {}
        # with torch.no_grad():
        #     p_r = self.model(p, inputs, bone_lengths=bone_lengths, sample=self.eval_sample, **kwargs)

        # occ_hat = p_r.view(batch_size, *shape)
        # voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        # for i in trange(batch_size):
        #     input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
        #     # vis.visualize_data(
        #     #     inputs[i].cpu(), self.input_type, input_img_path)
        #     vis.visualize_voxels(
        #         voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data, epoch_it):
        ''' Computes the loss.
        Args:
            data (dict): data dictionary
        '''
        device = self.device
        # print("device!!!!!", device)
        # p = data.get('points').to(device)
        # occ = data.get('occ').to(device)
        object_points = data.get('object_points').float().to(device)
        hand_joints = data.get('hand_joints').float().to(device)

        # import pdb; pdb.set_trace()
        # Use BPS
        if self.model.use_bps:
            object_points = data.get('object_bps').float().to(device)

        # import pdb; pdb.set_trace()

        kwargs = {}
        loss_dict = {}

        obj_c = self.model.encode_objects(object_points)
        # import pdb; pdb.set_trace()
        q_z = self.model.infer_z(hand_joints, obj_c, **kwargs)
        z = q_z.rsample()

        # Ignore object
        # c = c * 0.0
        # z = z * 0.0
        # print('z training', z)
        # print("device!!!!!", q_z)
        # print("device!!!!!", self.model.p0_z)

        # KL-divergence
        # if epoch_it > 0:
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss_kl = kl.mean()
        loss_dict['kl'] = loss_kl.item()
        # else:
        #     loss_kl = 0.0
        #     loss_dict['kl'] = 0.0
        # import pdb; pdb.set_trace()

        # joints
        pred = self.model.decode(z, obj_c, **kwargs)
        # print("pred", pred)
        if self.use_mano_loss:
            hand_verts = data.get('hand_verts').float().to(device)
            rot, pose, shape, trans = pred[:, :3], pred[:, 3:48], pred[:, 48:58], pred[:, 58:61]
            loss_hand_verts = self.mano_loss(rot, pose, shape, trans, hand_verts)
            loss_dict['vert'] = loss_hand_verts.item()
            loss = self.kl_weight * loss_kl + 1.0 * loss_hand_verts
            loss_dict['total'] = loss.item()
            return loss, loss_dict

        loss = self.mse_loss(pred, hand_joints)
        loss_dict['joints_recon'] = loss.item()

        # vis.visualise_skeleton(pred[0].detach().cpu().numpy(), object_points[0].detach().cpu().numpy(), show=True)
        # vis.visualise_skeleton(hand_joints[0].detach().cpu().numpy(), object_points[0].detach().cpu().numpy(), joint_order='mano', show=True)

        # Bone length loss
        loss_bone_length = self.bone_length_loss(pred, hand_joints)  # self.compute_bone_length_loss(pred, hand_joints)
        loss_dict['bone_length'] = loss_bone_length.item()

        # Root bone angle loss
        loss_root_bone_angle, loss_root_plane_angle = self.root_bone_angle_loss(pred, hand_joints)
        loss_dict['root_bone_angle'] = loss_root_bone_angle.item()
        loss_dict['root_plane_angle'] = loss_root_plane_angle.item()

        # All bone angle loss
        loss_all_bone_angle = self.all_bone_angle_loss(pred, hand_joints)
        loss_dict['all_bone_angle'] = loss_all_bone_angle.item()

        # Distance to object surface loss
        if self.use_surface_loss:
            gt_surface_dist = data.get('closest_point_dist').float().to(device)
            loss_surface_dist = self.surface_dist_loss(pred, object_points, gt_surface_dist)

        if self.model.use_bps:
            recon_weight = 100.0
            bone_weight = 1.0
        else:
            recon_weight = 2.0
            bone_weight = 2.0

        loss = (recon_weight * loss
                + bone_weight * loss_bone_length
                + 1.5 * loss_root_bone_angle  # 1.5
                + 1.5 * loss_root_plane_angle  # 1.5
                + 1.0 * loss_all_bone_angle  # 1.5
                + self.kl_weight * loss_kl
                )

        if self.use_surface_loss:
            # loss += 0.5 * loss_surface_dist
            loss_dict['surface_dist'] = loss_surface_dist.item()

        if self.use_refine_net:
            # import pdb; pdb.set_trace()
            tip_dists = data.get('tip_dists').float().to(device)
            noisy_joints = data.get('noisy_joints').float().to(device)
            refined_joints = self.model.refine_net(noisy_joints, obj_c, tip_dists)
            loss_refinement_l2 = self.mse_loss(refined_joints, hand_joints)
            loss_dict['refine_l2'] = loss_refinement_l2.item()

            loss_bone_length_re = self.bone_length_loss(refined_joints, hand_joints)
            loss_root_bone_angle_re, loss_root_plane_angle_re = self.root_bone_angle_loss(refined_joints, hand_joints)
            loss_all_bone_angle_re = self.all_bone_angle_loss(refined_joints, hand_joints)

            loss += (
                recon_weight * loss_refinement_l2
                + bone_weight * loss_bone_length_re
                + 1.5 * loss_root_bone_angle_re
                + 1.5 * loss_root_plane_angle_re  # 1.5
                + 1.0 * loss_all_bone_angle_re
            )
            loss_dict['refine_all'] = (
                loss_refinement_l2.item() + loss_bone_length_re.item() + loss_root_bone_angle_re.item()
                + loss_root_plane_angle_re.item() + loss_all_bone_angle_re.item()
            )

        if self.use_inter_loss:
            inside_points = data.get('inside_points').float().to(device)

            # from trimesh.base import Trimesh
            # tmp_joints = pred[None, 0]
            # output_mesh = self.model.halo_adapter(tmp_joints, joint_order='mano', original_position=True)
            # meshout_path = '/home/korrawe/halo_vae/exp/grab_refine_inter/test/hand.obj'
            # output_mesh.export(meshout_path)

            # # test query box
            # # inside_points = torch.rand(16, 4000, 3).cuda() - 0.5
            # # inside_points = inside_points * 25.

            # obj_points_tmp = inside_points[0].detach().cpu().numpy()
            # gt_object_points = Trimesh(vertices=obj_points_tmp)
            # obj_path = '/home/korrawe/halo_vae/exp/grab_refine_inter/test/obj.obj'
            # gt_object_points.export(obj_path)

            loss_inter, occ_p = self.inter_loss(pred, inside_points, self.model.halo_adapter)

            # one_idx = occ_p[0] > 0.5
            # obj_points_tmp = inside_points[0, one_idx].detach().cpu().numpy()
            # gt_object_points = Trimesh(vertices=obj_points_tmp)
            # obj_path = '/home/korrawe/halo_vae/exp/grab_refine_inter/test/intersect.obj'
            # gt_object_points.export(obj_path)
            # import pdb; pdb.set_trace()

            loss += 4.0 * loss_inter  # 5.0
            loss_dict['inter'] = loss_inter.item()

            # import pdb; pdb.set_trace()
            # grad_outputs = torch.ones_like(loss_inter)
            # grad = torch.autograd.grad(loss_inter, [pred], grad_outputs=grad_outputs, create_graph=True)[0]

        loss_dict['total'] = loss.item()
        return loss, loss_dict
