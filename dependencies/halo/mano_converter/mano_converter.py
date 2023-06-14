import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(0, "/home/korrawe/halo_vae")
from models.halo_adapter.converter import transform_to_canonical
from models.halo_adapter.interface import convert_joints, change_axes


def rot_mat_to_axis_angle(R):
    """
    Taken from
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/derivation/index.htm
    """
    # val_1 = (R[:,0,0] + R[:,1,1] + R[:,2,2] - 1) / 2
    # angles = torch.acos(val_1)
    # denom = 2 * torch.sqrt((val_1 ** 2 - 1).abs())
    # x = (R[:,2,1] - R[:,1,2]) / denom
    # y = (R[:,0,2] - R[:,2,0]) / denom
    # z = (R[:,1,0] - R[:,0,1]) / denom
    cos = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    cos = torch.clamp(cos, -1, 1)
    angles = torch.acos(cos)
    # angles = torch.acos((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2)
    denom = torch.sqrt(
            (R[..., 2, 1] - R[..., 1, 2]) ** 2 +
            (R[..., 0, 2] - R[..., 2, 0]) ** 2 +
            (R[..., 1, 0] - R[..., 0, 1]) ** 2
            )
    x = (R[..., 2, 1] - R[..., 1, 2]) / denom
    y = (R[..., 0, 2] - R[..., 2, 0]) / denom
    z = (R[..., 1, 0] - R[..., 0, 1]) / denom
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    axis = torch.cat((x, y, z), dim=-1)
    return axis, angles


def global_rot2mano_axisang(cano2pose_mat):

    global_rots = cano2pose_mat[:, :, :3, :3]
    R = global_rots

    # Convert to local rotation
    lvl1_idx = torch.tensor([1, 4, 7, 10, 13])
    lvl2_idx = torch.tensor([2, 5, 8, 11, 14])
    lvl3_idx = torch.tensor([3, 6, 9, 12, 15])
    R_lvl0 = R[:, None, 0]
    R_lvl1 = R[:, lvl1_idx]
    R_lvl2 = R[:, lvl2_idx]
    R_lvl3 = R[:, lvl3_idx]
    # Subtract rotation by parent bones
    R_lvl1_l = R_lvl0.transpose(-1, -2) @ R_lvl1
    R_lvl2_l = R_lvl1.transpose(-1, -2) @ R_lvl2
    R_lvl3_l = R_lvl2.transpose(-1, -2) @ R_lvl3
    # import pdb; pdb.set_trace()
    # R_local_no_order = torch.cat([R[:, None, 0], R_lvl1, R_lvl2_l, R_lvl3_l], dim=1)
    R_local_no_order = torch.cat([R[:, None, 0], R_lvl1_l, R_lvl2_l, R_lvl3_l], dim=1)
    # HALO order (also MANO internal order)
    reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
    R_local = R_local_no_order[:, reorder_idxs]
    # th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]

    axis, angles = rot_mat_to_axis_angle(R_local)  # global_rots
    # If root rotation is nan, set it to zero
    # Use torch.nan_to_num() for pytorch >1.8 https://pytorch.org/docs/1.8.1/generated/torch.nan_to_num.html
    # axis = torch.nan_to_num(axis)
    # print('axis before:', axis)
    axis[torch.isnan(axis)] = 0.
    # print('axis_after:', axis)
    # import pdb;pdb.set_trace()
    # axis[0, 0] = torch.tensor([0., 0., 0.])
    axis_angle_mano = axis * angles.unsqueeze(-1)
    return axis_angle_mano.reshape(axis_angle_mano.shape[0], -1)


# Shape computation-related
#
#
mano_2_zimm = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
zimm_2_ours = np.array([0, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20])
zimm_2_mano = np.argsort(mano_2_zimm)


def get_range(idx, len_range):
    n_idx = len(idx)
    idx = idx.repeat_interleave(len_range) * 3
    idx += torch.arange(len_range).repeat(n_idx)

    return idx


def initialize_QP(mano_layer, J):
    """
    For the QP problem:
    bT Q b + cT b + a = bl2

    returns Q, c, a, bl2
    """
    # Get T,S,M
    n_v = 778
    n_j = 16
    # Extract template mesh T and reshape from V x 3 to 3V
    T = mano_layer.th_v_template
    T = T.view(3 * n_v)
    # Extract Shape blend shapes and reshape from V x 3 x B to 3V x B
    S = mano_layer.th_shapedirs
    S = S.view(3 * n_v, 10)
    # Extract M and re-order to Zimmermann joint ordering.
    M = mano_layer.th_J_regressor
    # Add entries for the tips. TODO Add actual vertex positions
    M = torch.cat((M, torch.zeros((5, n_v)).to(J.device)), dim=0)
    # Convert to our joint ordering
    M = M[mano_2_zimm][zimm_2_ours]
    # Remove entries for tips. TODO Once using actual tip position, remove this step
    M = M[:16]
    # Construct the 3J x 3V band matrix
    M_band = torch.zeros(3 * n_j, 3 * n_v).to(J.device)
    fr = -(n_j - 1)
    to = n_v
    for i in range(fr, to):
        # Extract diagonal from M
        d = M.diag(i)
        # Expand it
        d = d.repeat_interleave(3)
        # Add it to the final band matrix
        M_band.diagonal(3 * i)[:] = d
    # Construct Q, c, a and bl for the quadratic equation: bT Q b + cT b + (a - bl2) = 0
    # Joint idx in Zimmermann ordering
    # idx_p = 10
    # idx_c = 11
    # Construct child/parent indices
    idx_p = torch.cat((torch.tensor([0]*5) , torch.arange(1,11)))
    idx_c = torch.arange(1,16)
    # Compute bl squared
    bl2 = torch.norm(J[idx_c] - J[idx_p], dim=-1).pow(2)

    idx_p_range = get_range(idx_p, 3)
    idx_c_range = get_range(idx_c, 3)

    M_c = M_band[idx_c_range]
    M_p = M_band[idx_p_range]
    # Exploit additional dimension to make it work across all bones
    M_c = M_c.view(15, 3, 3*n_v)
    M_p = M_p.view(15, 3, 3*n_v)
    T = T.view(1,3*n_v, 1)
    S = S.view(1,2334,10)
    bl2 = bl2.view(15,1,1)
    # Construct M_cp
    M_cp = (M_c - M_p).transpose(-1,-2) @ (M_c - M_p)
    # DEBUG
    # bone_idx = 0  # Should be root/thumb_mcp
    # M_c = M_c[bone_idx]
    # M_p = M_p[bone_idx]
    # M_cp = M_cp[bone_idx]
    # bl2 = bl2[bone_idx]
    # Compute a
    a = T.transpose(-1,-2) @ M_cp @ T
    # Compute c
    c = (
            S.transpose(-1,-2) @ (M_cp.transpose(-1,-2) @ T) + 
            S.transpose(-1,-2) @ (M_cp @ T)
        )
    # Compute Q
    Q = S.transpose(-1,-2) @ M_cp @ S

    return Q, c, a, bl2


def eval_QP(Q,c,a, bl2, b):
    b = b.view(1,10,1)
    val = ((b.transpose(-1,-2) @ Q @ b) + (c.transpose(-1,-2) @ b) + a)

    r = (val - bl2)

    return r


def get_J(Q,c, b):
    # Jacobian of QP problem
    J = Q @ b + c

    return J


def newtons_method(Q,c,a, bl2, beta_init, tol=1e-4):  # tol=1e-4):
    F = eval_QP(Q, c, a, bl2, beta_init)
    beta = beta_init.view(1,10,1)
    i = 0
    while F.abs().max() > tol:
        J = get_J(Q,c, beta)
        F = eval_QP(Q,c, a, bl2, beta)
        # Reshape matrices
        J = J.squeeze(-1)
        F = F.squeeze(-1)

        J_inv = (J.transpose(-1,-2) @ J).inverse() @ J.transpose(-1,-2)
        beta = beta - (J_inv @ F).unsqueeze(0)

        print(f'(Gauss-Newton) Max. residual: {F.abs().max()} mm')
        break  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        i += 1

    return beta, i
#
#
# END - Shape computation-related


class ManoConverter(nn.Module):
    def __init__(self, mano_layer, bmc_converter, device=None):
        super().__init__()
        self.mano_layer = mano_layer
        self.bmc_converter = bmc_converter
        self.device = device
        self.axisang_hand_mean = mano_layer.th_hands_mean

    def get_mano_shape(self, kps):
        '''Using Adrian's method for calculating shape parameter.
        Currently take in one hand at a time.
        '''
        J = convert_joints(kps, source='halo', target='biomech')
        J = J.squeeze(0)[:16]
        Q, c, a, bl2 = initialize_QP(self.mano_layer, J)
        # r = eval_QP(Q,c,a,bl2, b)
        # beta_init = torch.zeros_like(b)
        beta_init = torch.zeros(10).to(self.device)
        b_est, n_iter = newtons_method(Q, c, a, bl2, beta_init)
        # Construct MANO hand with estimated betas
        shape = b_est.view(1, 10)
        # import pdb; pdb.set_trace()

        # Defualt mean shape
        # shape = torch.zeros(1, 10).to(self.device)
        return shape

    def get_rest_joint(self, shape):
        rot = torch.zeros(1, 10).to(self.device)
        pose = torch.zeros(1, 45).to(self.device)
        pose_para = torch.cat([rot, pose], 1)
        _, _, _, _, rest_pose_joints, _ = self.mano_layer(pose_para, shape)
        return rest_pose_joints

    def _get_halo_matrices(self, trans_mat):
        # Use 16 out of 21 joints for nasa inputs
        joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])
        trans_mat = trans_mat[:, joints_for_nasa_input]
        return trans_mat

    def trans2bmc_cano_mat(self, joints):
        '''Assume 'halo' joint order and correct scale (m)
        '''
        # if joint_order != 'biomech':
        kps = convert_joints(joints, source='halo', target='biomech')
        # Assume right hand
        is_right_vec = torch.ones(kps.shape[0], device=self.device) * True

        # Global normalization - normalize index and middle finger root bone plane
        normalized_kps, normalization_mat = transform_to_canonical(kps, is_right=is_right_vec)
        normalized_kps, change_axes_mat = change_axes(normalized_kps, target='halo')
        normalization_mat = torch.matmul(change_axes_mat, normalization_mat)
        
        # import pdb; pdb.set_trace()
        # normalization_mat = torch.eye(4).to(self.device).unsqueeze(0)

        unpose_mat, _ = self.bmc_converter(normalized_kps, is_right_vec)
        # Change back to HALO joint order
        unpose_mat = convert_joints(unpose_mat, source='biomech', target='halo')
        unpose_mat = self._get_halo_matrices(unpose_mat)
        # unpose_mat_scaled = scale_halo_trans_mat(unpose_mat)

        full_trans_mat = torch.matmul(unpose_mat, normalization_mat)
        return full_trans_mat  # normalization_mat

    def get_trans_mat(self, rest_joints, hand_joints):
        # C
        mano_rest2bmc_cano_mat = self.trans2bmc_cano_mat(rest_joints)
        # B^-1
        posed_hand2bmc_cano_mat = self.trans2bmc_cano_mat(hand_joints)
        # BC
        mano_rest2posed_hand_mat = torch.matmul(torch.inverse(posed_hand2bmc_cano_mat), mano_rest2bmc_cano_mat)
        # import pdb;pdb.set_trace()
        return mano_rest2posed_hand_mat

    def remove_mean_pose(self, mano_input):
        # Subtract mean rotation from pose param
        mano_input = torch.cat([
            mano_input[:, :3],
            mano_input[:, 3:] - self.axisang_hand_mean
        ], 1)
        return mano_input

    def to_mano(self, kps):
        '''
        Take batch of key points in (m) as input .
        The keั points must follow HALO joint order.
        '''
        # Get shape param
        shape = self.get_mano_shape(kps)
        # From shape param, get mean pose and rest post (conditioned on the shape)
        rest_joints = self.get_rest_joint(shape)

        # Get trans formation matrices
        trans_mat = self.get_trans_mat(rest_joints, kps)

        # Convert trans_mat to axis-angle input for MANO
        mano_axisang = global_rot2mano_axisang(trans_mat)
        # Remove mean pose angle
        mano_pose = self.remove_mean_pose(mano_axisang)

        # shape = 0
        # pose = 0
        return shape, mano_pose

    def forward(self):
        pass
