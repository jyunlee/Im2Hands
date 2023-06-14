import numpy as np
import torch
import torch.nn as nn

from models.halo_adapter.converter import PoseConverter, transform_to_canonical, angle2, signed_angle
from models.halo_adapter.interface import convert_joints


def kp3D_to_bones(kp_3D, joint_parent, normalize_length=False):
    """
    Converts from joints to bones
    """
    eps_mat = torch.tensor(1e-9, device=kp_3D.device)
    batch_size = kp_3D.shape[0]
    bones = kp_3D[:, 1:] - kp_3D[:, joint_parent[1:]]  # .detach()
    if normalize_length:
        bone_lengths = torch.max(torch.norm(bones, dim=2, keepdim=True), eps_mat)
        # print("bone_length", bone_lengths)
        bones = bones / bone_lengths
    return bones


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def vis_bone(hand_joints, joint_parents):
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)

    b_start_loc = hand_joints[:, joint_parents[1:]][0]
    b_end_loc = hand_joints[:, 1:][0]
    for b in range(20):
        # color = 'r' if b in [1, 5, 9, 13, 17] else 'b'
        color = 'r' if b in [0, 4, 8, 12, 16] else 'b'
        ax.plot([b_start_loc[b, 0], b_end_loc[b, 0]],
                [b_start_loc[b, 1], b_end_loc[b, 1]],
                [b_start_loc[b, 2], b_end_loc[b, 2]], color=color)
    plt.show()


class BoneLengthLoss(nn.Module):
    def __init__(self, device=None, joint_parents=None):
        super().__init__()
        self.device = device
        if joint_parents is None:
            self.joint_parents = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, pred_joints, hand_joints):
        pred_bones = kp3D_to_bones(pred_joints, self.joint_parents)
        pred_bone_lengths = pred_bones.norm(dim=2)

        gt_bones = kp3D_to_bones(hand_joints, self.joint_parents)
        gt_bone_lengths = gt_bones.norm(dim=2)

        bone_length_loss = self.l2_loss(pred_bone_lengths, gt_bone_lengths)

        return bone_length_loss


def angle_diff(pred_angle, gt_angle):
    loss = torch.mean(
        torch.abs(torch.cos(pred_angle) - torch.cos(gt_angle)) +
        torch.abs(torch.sin(pred_angle) - torch.sin(gt_angle)),
    )
    return loss


class RootBoneAngleLoss(nn.Module):
    def __init__(self, device=None, joint_parents=None):
        super().__init__()
        self.device = device
        if joint_parents is None:
            self.joint_parents = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

        self.plane_angle_w = 1.0
        self.bone_angle_w = 1.0

    def forward(self, pred_joints, hand_joints):
        pred_bones = kp3D_to_bones(pred_joints, self.joint_parents, normalize_length=True)
        pred_angle = self._compute_root_bone_angle(pred_bones)
        pred_plane_angle = self._compute_root_plane_angle(pred_bones)

        gt_bones = kp3D_to_bones(hand_joints, self.joint_parents, normalize_length=True)
        gt_angle = self._compute_root_bone_angle(gt_bones)
        gt_plane_angle = self._compute_root_plane_angle(gt_bones)

        # vis_bone(hand_joints, self.joint_parents)
        # import pdb; pdb.set_trace()

        root_bone_angle_loss = angle_diff(pred_angle, gt_angle)
        root_plane_angle_loss = angle_diff(pred_plane_angle, gt_plane_angle)
        # import pdb; pdb.set_trace()

        return root_bone_angle_loss, root_plane_angle_loss

    def _compute_root_bone_angle(self, bones):
        """
        Assume MANO joint parent
        """
        # angle between (n0,n1), (n1,n2), (n2,n3)
        # thumb and index (plane n0)
        n0 = torch.cross(bones[:, 4], bones[:, 0])
        # middle and index (plane n1)
        n1 = torch.cross(bones[:, 8], bones[:, 4])
        # ring and middle (plane n2)
        n2 = torch.cross(bones[:, 12], bones[:, 8])
        # ring and pinky (plane n3)
        n3 = torch.cross(bones[:, 16], bones[:, 12])

        root_bone_angles = torch.stack([
            signed_angle(bones[:, 4], bones[:, 0], n0),
            signed_angle(bones[:, 8], bones[:, 4], n1),
            signed_angle(bones[:, 12], bones[:, 8], n2),
            signed_angle(bones[:, 16], bones[:, 12], n3)],
            dim=1
        )
        return root_bone_angles

    def _compute_root_plane_angle(self, bones):
        '''
        angles between root bone planes
        '''
        # angle between (n0,n1), (n1,n2), (n2,n3)
        # thumb and index (plane n0)
        n0 = torch.cross(bones[:, 4], bones[:, 0])
        # middle and index (plane n1)
        n1 = torch.cross(bones[:, 8], bones[:, 4])
        # ring and middle (plane n2)
        n2 = torch.cross(bones[:, 12], bones[:, 8])
        # ring and pinky (plane n3)
        n3 = torch.cross(bones[:, 16], bones[:, 12])

        root_plane_angles = torch.stack([
            signed_angle(n0, n1, bones[:, 4]),
            signed_angle(n2, n1, bones[:, 8]),
            signed_angle(n3, n2, bones[:, 12])],
            dim=1
        )
        return root_plane_angles


class AllBoneAngleLoss(nn.Module):
    def __init__(self, device=None, joint_parents=None):
        super().__init__()
        self.device = device
        if joint_parents is None:
            self.joint_parents = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

        # 3D keypoints to transformation matrices converter
        self.global_normalizer = transform_to_canonical
        self.pose_normalizer = PoseConverter(straight_hand=True)

    def forward(self, pred_joints, hand_joints):
        is_right_vec = torch.ones(pred_joints.shape[0], device=pred_joints.device)

        # vis_bone(hand_joints, self.joint_parents)

        hand_joints = convert_joints(hand_joints, source='mano', target='biomech')
        hand_joints_normalized, _ = self.global_normalizer(hand_joints, is_right=is_right_vec)
        gt_bone_angles = self.pose_normalizer(hand_joints_normalized, is_right_vec, return_rot_only=True)

        pred_joints = convert_joints(pred_joints, source='mano', target='biomech')
        pred_joints_normalized, _ = self.global_normalizer(pred_joints, is_right=is_right_vec)
        pred_bone_angles = self.pose_normalizer(pred_joints_normalized, is_right_vec, return_rot_only=True)

        all_bone_angle_loss = angle_diff(pred_bone_angles, gt_bone_angles)
        # all_bone_angle_loss = angle_diff(gt_bone_angles, gt_bone_angles)

        # import pdb; pdb.set_trace()

        return all_bone_angle_loss


class SurfaceDistanceLoss(nn.Module):
    def __init__(self, device=None, joint_parents=None):
        super().__init__()
        self.device = device
        # if joint_parents is None:
        #     self.joint_parents = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred_joints, object_points, gt_distance):
        # torch.cdist(a, b, p=2)
        pred_dist = torch.cdist(pred_joints, object_points)
        min_val, min_idx = torch.min(pred_dist, dim=2)

        surface_dist_loss = self.mse_loss(min_val, gt_distance)
        # import pdb; pdb.set_trace()
        return surface_dist_loss


class InterpenetrationLoss(nn.Module):
    def __init__(self, device=None, joint_parents=None):
        super().__init__()
        self.device = device
        self.joint_parents = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

    def forward(self, pred_joints, object_points, halo_model):
        # import pdb; pdb.set_trace()

        # object_points = object_points - pred_joints[:, None, 0]
        occ_p = halo_model.query_points(object_points, pred_joints, joint_order='mano')

        occ_p = torch.where(occ_p > 0.5, occ_p, torch.zeros_like(occ_p))
        penetration_loss = occ_p.mean()
        # import pdb; pdb.set_trace()
        # mask = (occ_p > 0.5).detach()
        # pred_dist = torch.cdist(pred_joints, object_points)
        # closest_dist, closest_joints = torch.min(pred_dist, dim=1)

        # #  Apply loss up along kinematic chain
        # loss_sum = 0

        return penetration_loss, occ_p


class ManoVertLoss(nn.Module):
    def __init__(self, device=None, joint_parents=None):
        super().__init__()
        self.device = device
        self.joint_parents = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

        self.mse_loss = torch.nn.MSELoss()

        import sys
        sys.path.insert(0, "/home/korrawe/halo_vae/scripts")
        from manopth.manolayer import ManoLayer
        from manopth import demo

        # import pdb; pdb.set_trace()
        self.mano_layer = ManoLayer(
            mano_root='/home/korrawe/halo_vae/scripts/mano/models', center_idx=0, use_pca=True, ncomps=45, flat_hand_mean=False)
        self.mano_layer = self.mano_layer.to(device)
        # hand_verts, hand_joints = mano_layer(torch.cat((rot, pose), 1), shape, trans)

    def forward(self, rot, pose, shape, trans, gt_hand_verts):
        # import pdb; pdb.set_trace()
        hand_verts, hand_joints = self.mano_layer(torch.cat((rot, pose), 1), shape, trans)
        hand_verts = hand_verts / 10.0
        hand_joints = hand_joints / 10.0
        vert_loss = self.mse_loss(hand_verts, gt_hand_verts)
        # occ_p = halo_model.query_points(object_points, pred_joints, joint_order='mano')

        # occ_p = torch.where(occ_p > 0.5, occ_p, torch.zeros_like(occ_p))
        # penetration_loss = occ_p.mean()
        # import pdb; pdb.set_trace()

        return vert_loss
