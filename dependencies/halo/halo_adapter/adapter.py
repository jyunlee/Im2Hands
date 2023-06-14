import sys
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules import loss

from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import (get_halo_model, convert_joints, change_axes,
                                           get_bone_lengths, scale_halo_trans_mat)
from models.halo_adapter.projection import get_projection_layer
from models.halo_adapter.transform_utils import xyz_to_xyz1
# from models.nasa_adapter.interface_helper

from models.utils import visualize as vis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HaloAdapter(nn.Module):
    def __init__(self, halo_config_file, store=False, device=None,
                 straight_hand=True, canonical_pose=None, is_right=True, denoiser_pth=None):
        super().__init__()
        self.device = device
        self.is_right = is_right

        # self.halo_model = get_halo_model(halo_config_file)
        self.halo_config_file = halo_config_file
        self.halo_model, self.halo_generator = get_halo_model(self.halo_config_file)
        # Freeze halo
        self.freeze_halo()

        # 3D keypoints to transformation matrices converter
        self.global_normalizer = transform_to_canonical
        self.pose_normalizer = PoseConverter(straight_hand=straight_hand)  # <- TODO: check this
        
        self.denoising_layer = None

    def freeze_halo(self):
        for param in self.halo_model.parameters():
            param.requires_grad = False

    def forward(self, joints, joint_order='biomech', return_kps=False, original_position=False):
        '''
        Input joints are in (cm)
        '''
        if joint_order != 'biomech':
            joints = convert_joints(joints, source=joint_order, target='biomech')

        halo_inputs, normalization_mat, normalized_kps = self.get_halo_inputs(joints)
        # print('in adaptor', normalization_mat)
        # import pdb; pdb.set_trace()

        ###
        nasa_mesh_out, stat = self.halo_generator.generate_mesh(halo_inputs)
        # # nasa_mesh_out.vertices = nasa_mesh_out.vertices * 0.4
        # # print(nasa_mesh_out)

        # nasa_out_file = os.path.join(args.out_folder, 'nasa_out.obj')
        # nasa_mesh_out.export(nasa_out_file)

        # nasa_cano_vertices = nasa_mesh_out.vertices.copy()

        # Return mesh in the original keypoint locations
        if original_position:
            # Check scale_halo_trans_mat() in interface.py for global scaling (0.4)
            mesh_verts = torch.Tensor(nasa_mesh_out.vertices).float().to(self.device) * 0.4
            ori_posi_verts = torch.matmul(torch.inverse(normalization_mat.double()), xyz_to_xyz1(mesh_verts).unsqueeze(-1).double()).squeeze(-1)

            # fig = plt.figure()
            # ax = fig.gca(projection=Axes3D.name)
            # fig.suptitle('red - before, blue - after', fontsize=16)
            # vis.plot_skeleton_single_view(joints.detach().cpu().numpy()[0], joint_order='biomech', color='r', ax=ax, show=False)
            # vis.plot_skeleton_single_view(normalized_kps.detach().cpu().numpy()[0], joint_order='biomech', color='b', ax=ax, show=False)
            # back_proj = torch.matmul(torch.inverse(normalization_mat), xyz_to_xyz1(normalized_kps).unsqueeze(-1)).squeeze(-1)
            # mesh_verts_np = mesh_verts.detach().cpu().numpy() * 100.0
            # ori_posi_verts_np = ori_posi_verts.detach().cpu().numpy() * 100.0
            # # ax.scatter(mesh_verts_np[:, 0], mesh_verts_np[:, 1], mesh_verts_np[:, 2], c='black', alpha=0.1)
            # ax.scatter(ori_posi_verts_np[:, 0], ori_posi_verts_np[:, 1], ori_posi_verts_np[:, 2], c='black', alpha=0.1)
            # # vis.plot_skeleton_single_view(back_proj.detach().cpu().numpy()[0] + 0.001, joint_order='biomech', color='orange', ax=ax, show=False)
            # fig.show()
            # import pdb; pdb.set_trace()

            # Scale back from m (HALO) to cm.
            nasa_mesh_out.vertices = ori_posi_verts[:, :3].detach().cpu().numpy() * 100.0
            # import pdb; pdb.set_trace()

        if not return_kps:
            return nasa_mesh_out

        ###
        # Undo normalization
        # import pdb; pdb.set_trace()
        undo_norm_kps = torch.matmul(torch.inverse(normalization_mat), xyz_to_xyz1(normalized_kps).unsqueeze(-1)).squeeze(-1)
        normalized_kps = undo_norm_kps
        ###
        return nasa_mesh_out, normalized_kps  # halo_outputs

    def query_points(self, query_points, joints, joint_order='biomech'):
        if joint_order != 'biomech':
            joints = convert_joints(joints, source=joint_order, target='biomech')

        scale = 100.0
        #scale = 40
        query_points = query_points / scale
        # query_points = query_points * 2.5
        # query_points, _ = change_axes(query_points, target='halo')

        halo_inputs, normalization_mat, normalized_kps = self.get_halo_inputs(joints)
        # import pdb; pdb.set_trace()

        query_points = torch.matmul(normalization_mat.unsqueeze(1), xyz_to_xyz1(query_points).unsqueeze(-1)).squeeze(-1)
        query_points = query_points[:, :, :3]
        query_points = query_points * 2.5

        occ_p = self.halo_model(query_points, halo_inputs['inputs'], bone_lengths=halo_inputs['bone_lengths'])
        return occ_p

    def get_halo_inputs(self, kps):
        '''
        This adapter globally normalize the input hand before computing the transformation matrices.
        The normalization matrix is "not" include in the HALO input.
        It is only for putting the mesh back to the position of the keypoints.
        Args:
            joints
        Returns:
            halo_input_dict:
            normalizeation_mat (in cm): Transformation matrices used to normalized the given pose to origin.
                Multiply the inverse of these matrices to go back to target pose.
        '''
        if not self.is_right:
            # TODO: Do left-to-right convertion
            pass

        is_right_vec = torch.ones(kps.shape[0], device=self.device) * self.is_right

        # Scale from cm (VAE) to m (HALO)
        scale = 100.0
        #scale = 1.0
        kps = kps / scale

        # Global normalization
        normalized_kps, normalization_mat = self.global_normalizer(kps, is_right=is_right_vec)

        # Denoise if available
        # Use HALO adapter to normalize middle root bone
        # target_js = convert_joints(target_js, source='mano', target='biomech')
        # target_js, unused_mat = transform_to_canonical(target_js, torch.ones(target_js.shape[0], device=device))
        # target_js = convert_joints(target_js, source='biomech', target='mano')
        # print(joints)
        if self.denoising_layer is not None:
            # Denoising layer operates in cm
            print("use denoiser in the forward pass")
            joints_before = normalized_kps.detach().cpu().numpy()[0]
            normalized_kps = self.denoising_layer(normalized_kps)
            joints_after = normalized_kps.detach().cpu().numpy()[0]

            fig = plt.figure()
            ax = fig.gca(projection=Axes3D.name)
            fig.suptitle('blue - before, orange - after', fontsize=16)
            vis.plot_skeleton_single_view(joints_before, joint_order='biomech', color='b', ax=ax, show=False)
            vis.plot_skeleton_single_view(joints_after, joint_order='biomech', color='orange', ax=ax, show=False)
            fig.show()

        normalized_kps, change_axes_mat = change_axes(normalized_kps, target='halo')
        normalization_mat = torch.matmul(change_axes_mat.double(), normalization_mat.double())

        # fig = plt.figure()
        # ax = fig.gca(projection=Axes3D.name)
        # fig.suptitle('red - before, blue - after', fontsize=16)
        # vis.plot_skeleton_single_view(kps.detach().cpu().numpy()[0], joint_order='biomech', color='r', ax=ax, show=False)
        # vis.plot_skeleton_single_view(normalized_kps.detach().cpu().numpy()[0], joint_order='biomech', color='b', ax=ax, show=False)
        # back_proj = torch.matmul(torch.inverse(normalization_mat), xyz_to_xyz1(normalized_kps).unsqueeze(-1)).squeeze(-1)
        # vis.plot_skeleton_single_view(back_proj.detach().cpu().numpy()[0] + 0.001, joint_order='biomech', color='orange', ax=ax, show=False)
        # fig.show()
        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        # # Scale from cm (VAE) to m (HALO)
        # scale = 100.0
        # normalized_kps = normalized_kps / scale

        bone_lengths = get_bone_lengths(normalized_kps, source='biomech', target='halo')
        # Compute unpose matrices
        unpose_mat, _ = self.pose_normalizer(normalized_kps, is_right_vec)
        # Test rotation numerical stability
        # import pdb; pdb.set_trace()
        # unpose_kps = torch.matmul(unpose_mat, xyz_to_xyz1(normalized_kps).unsqueeze(-1)).squeeze(-1)
        # vis.plot_skeleton_single_view(unpose_kps.detach().cpu().numpy()[0], joint_order='biomech')
        # cal_rotation_angles, _ = self.pose_normalizer(unpose_kps[..., :3], is_right_vec)
        # End test rotation

        # Change to HALO joint order
        unpose_mat = convert_joints(unpose_mat, source='biomech', target='halo')
        unpose_mat = self.get_halo_matrices(unpose_mat)
        unpose_mat_scaled = scale_halo_trans_mat(unpose_mat)

        halo_inputs = self._pack_halo_input(unpose_mat_scaled, bone_lengths)
        return halo_inputs, normalization_mat, normalized_kps * scale

    def get_halo_matrices(self, trans_mat):
        # Use 16 out of 21 joints for nasa inputs
        joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])
        trans_mat = trans_mat[:, joints_for_nasa_input]
        return trans_mat

    def _pack_halo_input(self, unpose_mat_scaled, bone_lengths):
        halo_inputs = {
            'inputs': unpose_mat_scaled,
            'bone_lengths': bone_lengths
        }
        return halo_inputs

    def optimize_trans(self, object_points, joints, gt_object_mesh, joint_order='mano'):
        epoch_coarse = 10
        # import pdb; pdb.set_trace()
        batch_size = joints.shape[0]
        trans = torch.zeros(batch_size, 3).to(self.device)
        trans.requires_grad_()

        fix_joints = joints.detach().clone()

        # # test query box
        # inside_points = torch.rand(1, 4000, 3).cuda() - 0.5
        # inside_points = inside_points * 12.

        # object_points = inside_points
        inside_points = object_points

        # from trimesh.base import Trimesh
        # obj_points_tmp = inside_points[0].detach().cpu().numpy()
        # gt_object_points = Trimesh(vertices=obj_points_tmp)
        # obj_path = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/test_optim/obj.obj'
        # gt_object_points.export(obj_path)

        # tmp_joints = fix_joints[None, 0]
        # output_mesh = self.forward(tmp_joints, joint_order='mano', original_position=True)
        # meshout_path = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/test_optim/hand_0.obj'
        # output_mesh.export(meshout_path)

        # occ_p = self.query_points(inside_points, tmp_joints, joint_order=joint_order)
        # one_idx = occ_p[0] > 0.5
        # obj_points_tmp = inside_points[0, one_idx].detach().cpu().numpy()
        # gt_object_points = Trimesh(vertices=obj_points_tmp)
        # obj_path = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/test_optim/intersect.obj'
        # gt_object_points.export(obj_path)

        # import pdb; pdb.set_trace()

        # Optimize for global translation (no trans)
        optimizer = torch.optim.Adam([trans], lr=0.1)  # trans

        for i in range(0, epoch_coarse):
            new_joints = fix_joints + trans
            occ_p = self.query_points(object_points, new_joints, joint_order=joint_order)

            occ_p = torch.where(occ_p > 0.5, occ_p, torch.zeros_like(occ_p))
            penetration_loss = occ_p.mean()

            # _, hand_joints = mano_layer(torch.cat((rot, pose), 1), shape, trans)
            # loss = criteria_loss(hand_joints, target_js)
            # print(loss)
            optimizer.zero_grad()
            penetration_loss.backward()
            optimizer.step()

            tmp_joints = fix_joints + trans

            # from trimesh.base import Trimesh
            # tmp_joints = pred[None, 0]
            # output_mesh = self.forward(tmp_joints, joint_order='mano', original_position=True)
            # meshout_path = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/test_optim/hand_%d.obj' % (i+1)
            # output_mesh.export(meshout_path)

            # print(' loss :', penetration_loss.item())
            # print(' trans :', trans)
            # pdb.set_trace()

        # new_joints = fix_joints + trans
        # output_mesh = self.forward(new_joints, joint_order='mano', original_position=True)
        # meshout_path = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/test_optim/hand_end.obj'
        # output_mesh.export(meshout_path)

        # obj_out_path = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/test_optim/obj_mesh.obj'
        # gt_object_mesh.export(obj_out_path)

        # print('After coarse alignment: %6f' % (penetration_loss.item()))

        # query_points(self, query_points, joints, joint_order='biomech')
        # import pdb; pdb.set_trace()
        return trans
