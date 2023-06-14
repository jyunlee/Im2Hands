# ------------------------------------------------------------------------------
# Copyright (c) 2019 Adrian Spurr
# Licensed under the GPL License.
# Written by Adrian Spurr
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dependencies.halo.halo_adapter.transform_utils import xyz_to_xyz1, pad34_to_44

eps = torch.tensor(1e-6)  # epsilon

def batch_dot_product(batch_1, batch_2, keepdim=False):
    """ Performs the batch-wise dot product
    """
    # n_elem = batch_1.size(1)
    # batch_size = batch_1.size(0)
    # # Idx of the diagonal
    # diag_idx = torch.tensor([[i,i] for i in range(n_elem)]).long()
    # # Perform for each element of batch a matmul
    # batch_prod = torch.matmul(batch_1, batch_2.transpose(2,1))
    # # Extract the diagonal
    # batch_dot_prod = batch_prod[:, diag_idx[:,0], diag_idx[:,1]]
    # if keepdim:
        # batch_dot_prod = batch_dot_prod.reshape(batch_size, -1, 1)

    batch_dot_prod = (batch_1 * batch_2).sum(-1, keepdim=keepdim)

    return batch_dot_prod

def rotate_axis_angle(v, k, theta):
    # Rotate v around k by theta using rodrigues rotation formula
    v_rot = v * torch.cos(theta) + \
            torch.cross(k,v.float())*torch.sin(theta) + \
            k*batch_dot_product(k,v, True)*(1-torch.cos(theta))

    return v_rot

def clip_values(x, min_v, max_v):
    clipped = torch.min(torch.max(x, min_v), max_v)

    return clipped

def pyt2np(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x

def normalize(bv, eps=1e-8):  # epsilon
    """
    Normalizes the last dimension of bv such that it has unit length in
    euclidean sense
    """
    eps_mat = torch.tensor(eps, device=bv.device)
    norm = torch.max(torch.norm(bv, dim=-1, keepdim=True), eps_mat)
    bv_n = bv / norm
    return bv_n

def angle2(v1, v2):
    """
    Numerically stable way of calculating angles.
    See: https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors
    """
    eps = 1e-10  # epsilon
    eps_mat = torch.tensor([eps], device=v1.device)
    n_v1 = v1 / torch.max(torch.norm(v1, dim=-1, keepdim=True), eps_mat)
    n_v2 = v2 / torch.max(torch.norm(v2, dim=-1, keepdim=True), eps_mat)
    a = 2 * torch.atan2(
        torch.norm(n_v1 - n_v2, dim=-1), torch.norm(n_v1 + n_v2, dim=-1)
    )
    return a

def signed_angle(v1, v2, ref):
    """
    Calculate signed angles of v1 with respect to v2

    The sign is positive if v1 x v2 points to the same direction as ref
    """
    def dot(x, y):
        return (x * y).sum(-1)

    angles = angle2(v1, v2)

    cross_v1_v2 = cross(v1, v2)
    # Compute sign
    cond = (dot(ref, cross_v1_v2) < 0).float()
    angles = cond * (-angles) + (1 - cond) * angles
    # import pdb; pdb.set_trace()
    return angles

def get_alignment_mat(v1, v2):
    """
    Returns the rotation matrix R, such that R*v1 points in the same direction as v2
    """
    axis = cross(v1, v2, do_normalize=True)
    ang = angle2(v1, v2)
    R = rotation_matrix(ang, axis)
    return R

def transform_to_canonical(kp3d, is_right, skeleton='bmc'):
    """Undo global translation and rotation
    """
    normalization_mat = compute_canonical_transform(kp3d.double(), is_right.double(), skeleton=skeleton)
    kp3d = xyz_to_xyz1(kp3d)
    # import pdb
    # pdb.set_trace()
    kp3d_canonical = torch.matmul(normalization_mat.unsqueeze(1), kp3d.unsqueeze(-1))
    kp3d_canonical = kp3d_canonical.squeeze(-1)
    # Pad T from 3x4 mat to 4x4 mat
    normalization_mat = pad34_to_44(normalization_mat)
    return kp3d_canonical, normalization_mat

def compute_canonical_transform(kp3d, is_right, skeleton='bmc'):
    """
    Returns a transformation matrix T which when applied to kp3d performs the following
    operations:
    1) Center at the root (kp3d[:,0])
    2) Rotate such that the middle root bone points towards the y-axis
    3) Rotates around the x-axis such that the YZ-projection of the normal of the plane
    spanned by middle and index root bone points towards the z-axis
    """
    assert len(kp3d.shape) == 3, "kp3d need to be BS x 21 x 3"
    assert is_right.shape[0] == kp3d.shape[0]
    is_right = is_right.type(torch.bool)
    dev = kp3d.device
    bs = kp3d.shape[0]
    kp3d = kp3d.clone().detach()
    # Flip so that we compute the correct transformations below
    kp3d[~is_right, :, 1] *= -1
    # Align root
    tx = kp3d[:, 0, 0]
    ty = kp3d[:, 0, 1]
    tz = kp3d[:, 0, 2]
    # Translation
    T_t = torch.zeros((bs, 3, 4), device=dev)
    T_t[:, 0, 3] = -tx
    T_t[:, 1, 3] = -ty
    T_t[:, 2, 3] = -tz
    T_t[:, 0, 0] = 1
    T_t[:, 1, 1] = 1
    T_t[:, 2, 2] = 1
    # Align middle root bone with -y-axis
    # x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(bs, 3)  # FIXME
    y_axis = torch.tensor([[0.0, -1.0, 0.0]], device=dev).expand(bs, 3)
    v_mrb = normalize(kp3d[:, 3] - kp3d[:, 0])
    R_1 = get_alignment_mat(v_mrb, y_axis)
    # Align x-y plane along plane spanned by index and middle root bone of the hand
    # after R_1 has been applied to it
    v_irb = normalize(kp3d[:, 2] - kp3d[:, 0])
    normal = cross(v_mrb, v_irb).view(-1, 1, 3)
    normal_rot = torch.matmul(normal, R_1.transpose(1, 2)).view(-1, 3)
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(bs, 3)
    R_2 = get_alignment_mat(normal_rot, z_axis)
    # Include the flipping into the transformation
    T_t[~is_right, 1, 1] = -1
    # Compute the canonical transform
    T = torch.bmm(R_2.double(), torch.bmm(R_1.double(), T_t.double()))
    return T


def set_equal_xyz_scale(ax, X, Y, Z):
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    return ax

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def plot_local_coord_system(local_cs, bones, bone_lengths, root, ax):
    local_cs = pyt2np(local_cs.squeeze())
    bones = pyt2np(bones.squeeze())
    bone_lengths = pyt2np(bone_lengths.squeeze())
    root = pyt2np(root.squeeze())
    col = ['r','g','b']
    n_fingers = 5
    n_b_per_f = 4
    print("local_cs", local_cs)
    for k in range(n_fingers):
        start_point = root.copy()
        for i in range(n_b_per_f):
            idx = i * n_fingers + k
            for j in range(3):
                ax.quiver(start_point[0], start_point[1], start_point[2],
                          local_cs[idx, j, 0], local_cs[idx, j, 1], local_cs[idx, j, 2],
                          color=col[j], length=10.1)
            old_start_point = start_point.copy()
            start_point += (bones[idx] * bone_lengths[idx])
            ax.plot([old_start_point[0], start_point[0]],
                    [old_start_point[1], start_point[1]],
                    [old_start_point[2], start_point[2]],
                    color="black")

    set_axes_equal(ax)
    plt.show()

def plot_local_coord(local_coords, bone_lengths, root, ax, show=True):
    local_coords = pyt2np(local_coords.squeeze())
    # bones = pyt2np(bones.squeeze())
    bone_lengths = pyt2np(bone_lengths.squeeze())
    root = pyt2np(root.squeeze())
    # root = root[0]
    print("root", root.shape)
    col = ['r', 'g', 'b']
    n_fingers = 5
    n_b_per_f = 4
    # print("local_cs", local_coords)
    # import pdb; pdb.set_trace()
    for k in range(n_fingers):
        start_point = root.copy()
        for i in range(n_b_per_f):
            idx = i * n_fingers + k
            # for j in range(3):
            # print("local_coords[idx]", local_coords[idx])
            local_bone = (local_coords[idx] * bone_lengths[idx])
            target_point = start_point + local_bone
            # print("start_point", start_point, "to", local_bone)
            cc = 'r' if show else 'b'
            ax.plot([start_point[0], target_point[0]],
                    [start_point[1], target_point[1]],
                    [start_point[2], target_point[2]],
                    color=cc)
            ax.scatter([target_point[0]], [target_point[1]], [target_point[2]], s=10.0)
            # start_point += (bones[idx] * bone_lengths[idx])
            start_point += (local_bone)
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if show:
        plt.show()


def rotation_matrix(angles, axis):
    """
    Converts Rodrigues rotation formula into a rotation matrix
    """
    eps = torch.tensor(1e-6)  # epsilon
    # print("norm", torch.abs(torch.sum(axis ** 2, dim=-1) - 1))
    try:
        assert torch.any(
            torch.abs(torch.sum(axis ** 2, dim=-1) - 1) < eps
        ), "axis must have unit norm"
    except:
        print("Warning: axis does not have unit norm")
        # import pdb
        # pdb.set_trace()
    dev = angles.device
    batch_size = angles.shape[0]
    sina = torch.sin(angles).view(batch_size, 1, 1)
    cosa_1_minus = (1 - torch.cos(angles)).view(batch_size, 1, 1)
    a_batch = axis.view(batch_size, 3)
    o = torch.zeros((batch_size, 1), device=dev)
    a0 = a_batch[:, 0:1]
    a1 = a_batch[:, 1:2]
    a2 = a_batch[:, 2:3]
    cprod = torch.cat((o, -a2, a1, a2, o, -a0, -a1, a0, o), 1).view(batch_size, 3, 3)
    I = torch.eye(3, device=dev).view(1, 3, 3)
    R1 = cprod * sina
    R2 = cprod.bmm(cprod) * cosa_1_minus
    R = I + R1 + R2
    return R


def cross(bv_1, bv_2, do_normalize=False):
    """
    Computes the cross product of the last dimension between bv_1 and bv_2.
    If normalize is True, it normalizes the vector to unit length.
    """
    cross_prod = torch.cross(bv_1.double(), bv_2.double(), dim=-1)
    if do_normalize:
        cross_prod = normalize(cross_prod)
    return cross_prod


def rotate(v, ax, rad):
    """
    Uses Rodrigues rotation formula
    Rotates the vectors in v around the axis in ax by rad radians. These 
    operations are applied on the last dim of the arguments. The parameter rad 
    is given in radian
    """
    # print("v", v, v.shape)
    # print("ax", ax, ax.shape)
    # print("rad", rad)
    sin = torch.sin
    cos = torch.cos
    v_rot = (
        v * cos(rad) + cross(ax, v) * sin(rad) + ax * batch_dot_product(ax, v, True) * (1 - cos(rad))
    )
    return v_rot


class PoseConverter(nn.Module):

    def __init__(self, store=False, dev='cpu', straight_hand=True, canonical_pose=None):
        super().__init__()

        # assert angle_poly.shape[0] == 15
        if not dev:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.store = store
        self.dev = dev
        # self.angle_poly = angle_poly.to(dev)

        self.idx_1 = torch.arange(1,21).to(dev).long()
        self.idx_2 = torch.zeros(20).to(dev).long()
        self.idx_2[:5] = 0
        self.idx_2[5:] = torch.arange(1,16)
        # For preprocess joints
        self.shift_factor = 0  # 0.5
        # For poly distance
        # n_poly = angle_poly.shape[0]
        # n_vert = angle_poly.shape[1]
        # idx_1 = torch.arange(n_vert)
        # idx_2 = idx_1.clone()
        # idx_2[:-1] = idx_1[1:]
        # idx_2[-1] = idx_1[0]
        # # n_poly x n_edges x 2
        # v1 = angle_poly[:,idx_1].to(dev)
        # v2 = angle_poly[:,idx_2].to(dev)
        # # n_poly x n_edges x 2
        # edges = (v2 - v1).view(1, n_poly, n_vert, 2)
        # Store
        # self.v1 = v1
        # self.v2 = v2
        # self.n_poly = n_poly
        # self.n_vert = n_vert
        # self.edges = edges
        self.dot = lambda x,y: (x*y).sum(-1)
        # self.l2 = self.dot(edges,edges)
        self.zero = torch.zeros((1), device=dev)
        self.one = torch.ones((1), device=dev)
        # For angle computation
        self.rb_idx = torch.arange(5, device=dev).long()
        self.nrb_idx_list = []
        for i in range(2, 4):
            self.nrb_idx_list += [torch.arange(i*5, (i+1)*5, device=dev).long()]
        self.nrb_idx = torch.arange(5,20, device=dev).long()
        self.one = torch.ones((1), device=dev)
        self.zero = torch.zeros((1), device=dev)
        self.eps_mat = torch.tensor(1e-9, device=dev)  # epsilon
        self.eps = eps
        self.eps_poly = 1e-2

        self.y_axis = torch.tensor([[[0,1.,0]]], device=dev)
        self.x_axis = torch.tensor([[[1.,0,0]]], device=dev)
        self.z_axis = torch.tensor([[[0],[0],[1.]]], device=dev)
        self.xz_mat = torch.tensor([[[[1.,0,0],[0,0,0],[0,0,1]]]], device=dev)
        self.yz_mat = torch.tensor([[[[0,0,0],[0,1.,0],[0,0,1]]]], device=dev)
        self.flipLR = torch.tensor([[[-1.,1.,1.]]], device=dev)

        self.bones = None
        self.bone_lengths = None
        self.local_cs = None
        self.local_coords = None
        self.rot_angles = None

        if canonical_pose is not None:
            self.initialize_canonical_pose(canonical_pose)
        else:
            # initialize canonical pose with some value
            self.root_plane_angles = np.array([0.8, 0.2, 0.2])  # np.array([0.0, 0.0, 0.0])
            self.root_bone_angles = np.array([0.4, 0.2, 0.2, 0.2])

        if straight_hand:
            # For NASA - no additional rotation
            self.canonical_rot_angles = torch.tensor([[[0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]]]
            )
        else:
            # # For MANO
            self.canonical_rot_angles = torch.tensor([[[-6.8360e-01,  2.8175e-01],
                [-1.3016e+00, -1.4236e-03],
                [-1.5708e+00, -1.4236e-03],
                [-1.8425e+00,  7.4600e-02],
                [-2.0746e+00,  1.8700e-01],
                [-4.2529e-01,  1.4156e-01],
                [-1.5473e-01,  1.8678e-01],
                [-7.1449e-02,  1.4358e-01],
                [-8.7801e-02, -1.5822e-01],
                [-1.4013e-01,  3.4336e-02],
                [ 3.3824e-01,  1.6999e-01],
                [ 1.8830e-01,  4.8844e-02],
                [ 1.1238e-01, -8.7551e-03],
                [ 1.1125e-01,  1.6023e-01],
                [ 5.3791e-02, -4.2887e-02],
                [-1.0314e-01, -1.2587e-02],
                [-9.8003e-02, -1.7479e-01],
                [-6.6223e-02,  2.9800e-02],
                [-2.4101e-01, -5.5725e-02],
                [-1.3926e-01, -8.2840e-02]]]
            )

    def initialize_canonical_pose(self, canonical_joints):
        # canonical_joints must be of size [1, 21]
        # Pre-process the joints
        joints = self.preprocess_joints(canonical_joints, torch.ones(canonical_joints.shape[0], device=canonical_joints.device))
        # Compute the bone vectors
        bones, bone_lengths, kp_to_bone_mat = self.kp3D_to_bones(joints)

        self.root_plane_angles = self._compute_root_plane_angle(bones)  # np.array([0.8, 0.2, 0.2])
        self.root_bone_angles = self._compute_root_bone_angle(bones)  # np.array([0.4, 0.2, 0.2, 0.2])

        # Compute the local coordinate systems for each bone
        # This assume the root bones are fixed
        local_cs = self.compute_local_coordinate_system(bones)
        # Compute the local coordinates
        local_coords = self.compute_local_coordinates(bones, local_cs)

        # Copmute the rotation around the y and rotated-x axis
        self.canonical_rot_angles = self.compute_rot_angles(local_coords)
        # check dimentions
        import pdb; pdb.set_trace()
        pass

    def _compute_root_plane_angle(self, bones):
        root_plane_angle = np.zeros(3)
        # canonical_angle defines angles between root bone planes
        # angle between (n0,n1), (n1,n2), (n2,n3)
        # middle and ring (plane n2)
        n2 = torch.cross(bones[:, 3], bones[:, 2])
        # index and middle (plane n1)
        n1 = torch.cross(bones[:, 2], bones[:, 1])
        root_plane_angle[1] = angle2(n1, n2).squeeze(0)
        # thumb and index (plane n0)
        n0 = torch.cross(bones[:, 1], bones[:, 0])
        root_plane_angle[0] = angle2(n0, n1).squeeze(0)
        # ring and pinky (plane n4)
        n3 = torch.cross(bones[:, 4], bones[:, 3])
        root_plane_angle[2] = angle2(n2, n3).squeeze(0)
        return root_plane_angle

    def _compute_root_bone_angle(self, bones):
        root_bone_angles = np.zeros(4)
        root_bone_angles[0] = angle2(bones[:, 1], bones[:, 0]).squeeze(0)
        root_bone_angles[1] = angle2(bones[:, 2], bones[:, 1]).squeeze(0)
        root_bone_angles[2] = angle2(bones[:, 3], bones[:, 2]).squeeze(0)
        root_bone_angles[3] = angle2(bones[:, 4], bones[:, 3]).squeeze(0)
        return root_bone_angles

    def kp3D_to_bones(self, kp_3D):
        """
        Converts from joints to bones
        """
        dev = self.dev
        eps_mat = self.eps_mat
        batch_size = kp_3D.shape[0]
        bones = kp_3D[:, self.idx_1] - kp_3D[:, self.idx_2]  # .detach()
        bone_lengths = torch.max(torch.norm(bones, dim=2, keepdim=True), eps_mat)
        bones = bones / bone_lengths

        translate = torch.eye(4, device=dev).repeat(batch_size, 20, 1, 1)
        # print("translate", translate.shape)
        # print("kp 3D", kp_3D.shape)
        translate[:, :20, :3, 3] = -1. * kp_3D[:, self.idx_2]
        # print(translate.detach())
        scale = torch.eye(4, device=dev).repeat(batch_size, 20, 1, 1)
        # print("bone_lengths", bone_lengths.shape)
        scale = scale * 1. / bone_lengths.unsqueeze(-1)
        scale[:, :, 3, 3] = 1.0
        # print("scale", scale)

        kp_to_bone_mat = torch.matmul(scale.double(), translate.double())
        # print(kp_to_bone_mat)
        # import pdb;pdb.set_trace()
        # assert False
        return bones, bone_lengths, kp_to_bone_mat

    def compute_bone_to_kp_mat(self, bone_lengths, local_coords_canonical):
        bone_to_kp_mat = torch.eye(4, device=bone_lengths.device).repeat(*bone_lengths.shape[:2], 1, 1)
        # scale
        bone_to_kp_mat = bone_to_kp_mat * bone_lengths.unsqueeze(-1)
        bone_to_kp_mat[:, :, 3, 3] = 1.0
        # print("bone_to_kp_mat", bone_to_kp_mat, bone_to_kp_mat.shape)
        # assert False

        # add translation along kinematic chain
        # no root
        lev_1 = [0, 1, 2, 3, 4]
        lev_2 = [5, 6, 7, 8, 9]
        lev_3 = [10, 11, 12, 13, 14]
        lev_4 = [15, 16, 17, 18, 19]

        bones_scaled = local_coords_canonical * bone_lengths
        # print("bone length shape", bone_lengths.shape)

        lev_1_trans = torch.zeros([bone_lengths.shape[0], 5, 3], device=bone_lengths.device)
        lev_2_trans = bones_scaled[:, lev_1]
        lev_3_trans = bones_scaled[:, lev_2] + lev_2_trans
        lev_4_trans = bones_scaled[:, lev_3] + lev_3_trans

        translation = torch.cat([lev_1_trans, lev_2_trans, lev_3_trans, lev_4_trans], dim=1)
        # print("translation", translation.shape)

        bone_to_kp_mat[:, :, :3, 3] = translation  # [:, :, :]
        # print(bone_to_kp_mat)
        # assert False

        return bone_to_kp_mat

    def compute_local_coordinate_system(self, bones):
        dev = self.dev
        dot = batch_dot_product
        n_fingers = 5
        batch_size = bones.size(0)
        n_bones = bones.size(1)
        rb_idx = self.rb_idx
        # rb_idx_bin = self.rb_idx_bin
        nrb_idx_list = self.nrb_idx_list
        one = self.one
        eps = self.eps
        eps_mat = self.eps_mat
        xz_mat = self.xz_mat
        z_axis = self.z_axis
        bs = bones.size(0)
        y_axis = self.y_axis.repeat(bs, 5, 1)
        x_axis = self.x_axis.repeat(bs, 5, 1)
        # Get the root bones
        root_bones = bones[:, rb_idx]
        # Compute the plane normals for each neighbouring root bone pair
        # Compute the plane normals directly
        plane_normals = torch.cross(root_bones[:,:-1], root_bones[:,1:], dim=2)
        # Compute the plane normals flipped (sometimes gives better grad)
        # WARNING: Uncomment flipping below
        # plane_normals = torch.cross(root_bones[:,1:], root_bones[:,:-1], dim=2)
        # Normalize them
        plane_norms = torch.norm(plane_normals, dim=2, keepdim=True)
        plane_norms = torch.max(plane_norms, eps_mat)
        plane_normals = plane_normals / plane_norms
        # Define the normals of the planes on which the fingers reside (model assump.)
        finger_plane_norms = torch.zeros((batch_size, n_fingers, 3), device=dev)
        finger_plane_norms[:,0] = plane_normals[:,0]
        finger_plane_norms[:,1] = plane_normals[:,1]
        finger_plane_norms[:,2] = (plane_normals[:,1] + plane_normals[:,2]) / 2
        finger_plane_norms[:,3] = (plane_normals[:,2] + plane_normals[:,3]) / 2
        finger_plane_norms[:,4] = plane_normals[:,3]
        # Flip the normals s.t they look towards the palm of the hands
        # finger_plane_norms = -finger_plane_norms
        # Root bones are in the global coordinate system
        coord_systems = torch.zeros((batch_size, n_bones, 3, 3), device=dev)
        # Root bone coordinate systems
        coord_systems[:, rb_idx] = torch.eye(3, device=dev)
        # Root child bone coordinate systems
        z = bones[:, rb_idx]
        y = torch.cross(bones[:,rb_idx].double(), finger_plane_norms.double())
        x = torch.cross(y,z)
        # Normalize to unit length
        x_norm = torch.max(torch.norm(x, dim=2, keepdim=True), eps_mat)
        x = x / x_norm
        y_norm = torch.max(torch.norm(y, dim=2, keepdim=True), eps_mat)
        y = y / y_norm
        # Parent bone is already normalized
        # z = z / torch.norm(z, dim=2, keepdim=True)
        # Assign them to the coordinate system
        coord_systems[:, rb_idx + 5, 0] = x.float()
        coord_systems[:, rb_idx + 5, 1] = y.float()
        coord_systems[:, rb_idx + 5, 2] = z.float()
        # Construct the remaining bone coordinate systems iteratively
        # TODO This can be potentially sped up by rotating the root child bone
        # instead of the parent bone
        for i in range(2, 4):
            idx = nrb_idx_list[i - 2]
            bone_vec_grandparent = bones[:, idx - 2 * 5]
            bone_vec_parent = bones[:, idx - 1 * 5]
            # bone_vec_child = bones[:,idx]
            p_coord = coord_systems[:, idx - 1 * 5]
            ###### IF BONES ARE STRAIGHT LINE
            # Transform into local coordinates
            lbv_1 = torch.matmul(p_coord.float(), bone_vec_grandparent.unsqueeze(-1).float())
            lbv_2 = torch.matmul(p_coord.float(), bone_vec_parent.unsqueeze(-1).float())
            ###### Angle_xz
            # Project onto local xz plane
            lbv_2_xz = torch.matmul(xz_mat, lbv_2).squeeze(-1)
            lbv_2 = lbv_2.squeeze(-1)
            # Compute the dot product
            dot_prod_xz = torch.matmul(lbv_2_xz, z_axis).squeeze(-1)
            # If dot product is close to zero, set it to zero
            cond_0 = (torch.abs(dot_prod_xz) < 1e-6).float()
            dot_prod_xz = cond_0 * 0 + (1 - cond_0) * dot_prod_xz
            # Compute the norm and make sure its non-zero
            norm_xz = torch.max(torch.norm(lbv_2_xz, dim=-1), eps_mat)
            # Normalize the dot product
            dot_prod_xz = dot_prod_xz / norm_xz
            # Clip such that we do not get NaNs during GD
            dot_prod_xz = clip_values(dot_prod_xz, -one+eps, one-eps)
            # Compute the angle from the z-axis
            angle_xz = torch.acos(dot_prod_xz)
            # If lbv2_xz is on the -x side, we interpret it as -angle
            cond_1 = ((lbv_2_xz[:,:,0] + 1e-6) < 0).float()
            angle_xz = cond_1 * (-angle_xz) + (1-cond_1) * angle_xz
            ###### Angle_yz
            # Compute the normalized dot product
            dot_prod_yz = batch_dot_product(lbv_2_xz, lbv_2).squeeze(-1)
            dot_prod_yz = dot_prod_yz / norm_xz
            dot_prod_yz = clip_values(dot_prod_yz, -one+eps, one-eps)
            # Compute the angle from the projected bone
            angle_yz = torch.acos(dot_prod_yz)
            # If bone is on -y side, we interpret it as -angle
            cond_2 = ((lbv_2[:,:,1] + 1e-6) < 0).float()
            angle_yz = cond_2 * (-angle_yz) + (1-cond_2) * angle_yz
            ###### Compute the local coordinate system
            angle_xz = angle_xz.unsqueeze(-1)
            angle_yz = angle_yz.unsqueeze(-1)
            # Transform rotation axis to global
            rot_axis_xz = torch.matmul(p_coord.transpose(2,3),
                    y_axis.unsqueeze(-1))
            rot_axis_y = rotate_axis_angle(x_axis, y_axis, angle_xz)
            rot_axis_y = torch.matmul(p_coord.transpose(2,3),
                    rot_axis_y.unsqueeze(-1))
            rot_axis_y = rot_axis_y.squeeze(-1)
            rot_axis_xz = rot_axis_xz.squeeze(-1)
            
            cond = (torch.abs(angle_xz) < eps).float()
            x = cond*x + (1-cond)*rotate_axis_angle(x, rot_axis_xz, angle_xz)
            y = cond*y + (1-cond)*rotate_axis_angle(y, rot_axis_xz, angle_xz)
            z = cond*z + (1-cond)*rotate_axis_angle(z, rot_axis_xz, angle_xz)
            # Rotate around rotated x/-x
            cond = (torch.abs(angle_yz) < eps).float()
            x = cond*x + (1-cond)*rotate_axis_angle(x, rot_axis_y, -angle_yz)
            y = cond*y + (1-cond)*rotate_axis_angle(y, rot_axis_y, -angle_yz)
            z = cond*z + (1-cond)*rotate_axis_angle(z, rot_axis_y, -angle_yz)

            coord_systems[:, idx, 0] = x.float()
            coord_systems[:, idx, 1] = y.float()
            coord_systems[:, idx, 2] = z.float()

        return coord_systems.detach()


    def compute_local_coordinates(self, bones, coord_systems):
        local_coords = torch.matmul(coord_systems, bones.unsqueeze(-1))

        return local_coords.squeeze(-1)


    def compute_rot_angles(self, local_coords):
        n_bones = local_coords.size(1)
        z_axis = self.z_axis
        xz_mat = self.xz_mat
        yz_mat = self.yz_mat
        one = self.one
        eps = self.eps
        eps_mat = self.eps_mat
        # Compute the flexion angle
        # Project bone onto the xz-plane
        proj_xz = torch.matmul(xz_mat, local_coords.unsqueeze(-1)).squeeze(-1)
        norm_xz = torch.max(torch.norm(proj_xz, dim=-1), eps_mat)
        dot_prod_xz = torch.matmul(proj_xz, z_axis).squeeze(-1)
        cond_0 = (torch.abs(dot_prod_xz) < 1e-6).float()
        dot_prod_xz = cond_0 * 0 + (1-cond_0) * dot_prod_xz
        dot_prod_xz = dot_prod_xz / norm_xz
        dot_prod_xz = clip_values(dot_prod_xz, -one+eps, one-eps)
        # Compute the angle from the z-axis
        angle_xz = torch.acos(dot_prod_xz)
        # If proj_xz is on the -x side, we interpret it as -angle
        cond_1 = ((proj_xz[:,:,0] + 1e-6) < 0).float()
        angle_xz = cond_1 * (-angle_xz) + (1-cond_1) * angle_xz
        # Compute the abduction angle 
        dot_prod_yz = batch_dot_product(proj_xz, local_coords).squeeze(-1)
        dot_prod_yz = dot_prod_yz / norm_xz
        dot_prod_yz = clip_values(dot_prod_yz, -one+eps, one-eps)
        # Compute the angle from the projected bone
        angle_yz = torch.acos(dot_prod_yz)
        # If bone is on y side, we interpret it as -angle
        cond_2 = ((local_coords[:,:,1] + 1e-6) > 0).float()
        angle_yz = cond_2 * (-angle_yz) + (1-cond_2) * angle_yz
        # Concatenate both matrices
        rot_angles = torch.cat((angle_xz.unsqueeze(-1), angle_yz.unsqueeze(-1)),
                dim=-1)

        return rot_angles


    def preprocess_joints(self, joints, is_right):
        """
        This function does the following:
        - Move palm-centered root to wrist-centered root
        - Root-center (for easier flipping)
        - Flip left hands to right
        """
        # Had to formulate it this way such that backprop works
        joints_pp = 0 + joints
        # Vector from palm to wrist (simplified expression on paper)
        vec = joints[:,0] - joints[:,3]
        vec = vec / torch.norm(vec, dim=1, keepdim=True)
        # This is BUG !!!!
        # Shift palm in direction wrist with factor shift_factor
        joints_pp[:,0] = joints[:,0] + self.shift_factor * vec

        # joints_pp[:,0] = 2*joints[:,0] - joints[:,3]
        # joints_pp = joints_pp - joints_pp[:,0]
        # if not kp3d_is_right:
            # joints_pp = joints_pp * torch.tensor([[-1.,1.,1.]]).view(-1,1,3)

        # Flip left handed joints
        is_right = is_right.view(-1,1,1)
        joints_pp = joints_pp * is_right + (1-is_right) * joints_pp * self.flipLR
        # DEBUG
        # joints = joints.clone()
        # palm = joints[:,0]
        # middle_mcp = joints[:,3]
        # wrist = 2*palm - middle_mcp
        # joints[:,0] = wrist
        # # Root-center (for easier flipping)
        # joints = joints - joints[:,0]
        # # Flip if left hand
        # if not kp3d_is_right:
            # joints[:,:,0] = joints[:,:,0] * -1

        # import pdb;pdb.set_trace()
        return joints_pp


    # def polygon_distance(self, angles):
    #     """
    #     Computes the distance of p_b[:,i] to polys[i]
    #     """
    #     # Slack variable due to numerical imprecision
    #     eps = self.eps_poly
    #     # Batch-dot prod
    #     dot = self.dot
    #     polys = self.angle_poly
    #     n_poly = self.n_poly
    #     n_vert = self.n_vert
    #     v1 = self.v1
    #     v2 = self.v2
    #     edges = self.edges
    #     zero = self.zero
    #     one = self.one
    #     l2 = self.l2
    #     # Check if the polygon contains the point
    #     # batch_size x n_poly x n_edges x 2
    #     # Distance of P[:,i] to all of poly[i] edges
    #     line = angles.view(-1,n_poly,1,2) - v1.view(1,n_poly,n_vert,2)
    #     # n_points x n_vertices x 1
    #     cross_prod = edges[:,:,:,0]*line[:,:,:,1] - edges[:,:,:,1]*line[:,:,:,0]
    #     # Reduce along the n_vertices dim
    #     contains = (cross_prod >= -eps)
    #     contains = contains.sum(dim=-1)
    #     contains = (contains==n_vert).float()

    #     t = torch.max(zero, torch.min(one, dot(edges,line) / l2)).unsqueeze(-1)
    #     proj = v1 + t * edges

    #     angles = angles.view(-1, n_poly,1,2)
    #     # Compute distance over all vertices
    #     d = torch.sum(
    #             torch.abs(torch.cos(angles) - torch.cos(proj)) + 
    #             torch.abs(torch.sin(angles) - torch.sin(proj)),
    #             dim=-1)
    #     # Get the min
    #     D, _ = torch.min(d, dim=-1)
    #     # Assign 0 for points that are contained in the polygon
    #     d = (contains * 0 + (1-contains) * D) ** 2

    #     return d
    
    def compute_rotation_matrix(self, rot_angles, bone_local):
        ''' rot_angles [BS, bone, 2 (flexion angle, abduction angle)]
        '''
        batch_size, bone, xy_size = rot_angles.shape
        rot_angles_flat = rot_angles.reshape(batch_size * bone, 2)
        bone_local_flat = bone_local.reshape(batch_size * bone, 3)

        # mano canonical pose
        canonical_rot_flat = self.canonical_rot_angles.repeat(batch_size, 1, 1).to(rot_angles_flat.device)
        canonical_rot_flat = canonical_rot_flat.reshape(batch_size * bone, 2)

        x = torch.zeros([batch_size * bone, 3], device=rot_angles_flat.device)
        y = torch.zeros([batch_size * bone, 3], device=rot_angles_flat.device)
        z = torch.zeros([batch_size * bone, 3], device=rot_angles_flat.device)
        x[:, 0] = 1.
        y[:, 1] = 1.
        z[:, 2] = 1.

        rotated_x = rotate(x, y, rot_angles_flat[:, 0].unsqueeze(1))
        # print("rotated x", rotated_x, rotated_x.shape)

        # print("bone local", bone_local_flat, bone_local_flat.shape)

        # reverse transform starts here
        # abduction
        b_local_1 = rotate(bone_local_flat, rotated_x, -rot_angles_flat[:, 1].unsqueeze(1))
        # flexion
        b_local_2 = rotate(b_local_1, y, -rot_angles_flat[:, 0].unsqueeze(1))

        # print("sanity check", (b_local_2 - z).abs().max())
        # assert (b_local_2 - z).abs().max() < torch.tensor(1e-5)

        abduction_angle = (-rot_angles_flat[:, 1] + canonical_rot_flat[:, 1]).unsqueeze(1)
        # abduction_angle = (-rot_angles_flat[:, 1]).unsqueeze(1)
        r_1 = rotation_matrix(abduction_angle, rotated_x)
        flexion_angle = (-rot_angles_flat[:, 0] + canonical_rot_flat[:, 0]).unsqueeze(1)
        # flexion_angle = (-rot_angles_flat[:, 0]).unsqueeze(1)
        r_2 = rotation_matrix(flexion_angle, y)
        # print("abduction angle", abduction_angle.shape)
        # assert False
        
        # print("r_1", r_1.shape)
        # print("r_2", r_2.shape)
        r = 0
        r = torch.bmm(r_2.float(), r_1.float())
        r = r.reshape(batch_size, bone, 3, 3)

        # mask root bones rotation
        r[:, :5] = torch.eye(3, device=r.device)
        # print("final r", r)

        # x_angle = rot_angles[:, :, 0]
        # y_angle = rot_angles[:, :, 1]

        # print("rot_angles", rot_angles.shape)
        # print("x_angle", x_angle.shape)
        # print("y_angle", y_angle.shape)

        # rot_x_mat = get_rot_mat_x(x_angle)
        # rot_y_mat = get_rot_mat_y(y_angle)

        # rot_x_y = torch.matmul(rot_y_mat, rot_x_mat)
        return r #  rot_x_y


    def get_scale_mat_from_bone_lengths(self, bone_lengths):
        scale_mat = torch.eye(3, device=bone_lengths.device).repeat(*bone_lengths.shape[:2], 1, 1)
        # print("eye", scale_mat, scale_mat.shape)
        scale_mat = bone_lengths.unsqueeze(-1) * scale_mat
        # print("scale_mat", scale_mat, scale_mat.shape)

        return scale_mat

    
    def get_trans_mat_with_translation(self, trans_mat_without_scale_translation, local_coords_after_unpose, bones, bone_lengths):
        # print("---- get trans mat with translation -----")
        # print("trans_mat_without_scale_translation", trans_mat_without_scale_translation.shape)
        # print("bones", bones.shape)
        # print("bone_lengths", bone_lengths.shape)
        translation = local_coords_after_unpose * bone_lengths
        # translation = translation.unsqueeze(-1)

        # print("translation", translation.shape)
        # print(translation)

        # add translation along kinematic chain
        # no root
        lev_1 = [0, 1, 2, 3, 4]
        lev_2 = [5, 6, 7, 8, 9]
        lev_3 = [10, 11, 12, 13, 14]
        lev_4 = [15, 16, 17, 18, 19]

        root_trans =  translation[:, lev_1] * 0.
        lev_1_trans = translation[:, lev_1]
        lev_2_trans = translation[:, lev_2] + lev_1_trans
        lev_3_trans = translation[:, lev_3] + lev_2_trans
        lev_4_trans = translation[:, lev_4] + lev_3_trans

        # print("lev_1_trans", lev_1_trans)
        # print("lev_3_trans", lev_3_trans, lev_3_trans.shape)

        # final_trans = torch.cat([lev_1_trans, lev_2_trans, lev_3_trans, lev_4_trans], dim=1)
        final_trans = torch.cat([root_trans, lev_1_trans, lev_2_trans, lev_3_trans], dim=1)
        # print("final trans", final_trans, final_trans.shape)
        final_trans = final_trans.unsqueeze(-1)


        trans_mat = torch.cat([trans_mat_without_scale_translation, final_trans], dim=3)
        # print("trans_mat", trans_mat.shape)
        last_row = torch.tensor([0., 0., 0., 1.], device=trans_mat.device).repeat(*trans_mat.shape[:2], 1 , 1)
        trans_mat = torch.cat([trans_mat, last_row], dim=2)
        # start_point = (bones[idx] * bone_lengths[idx])
        # print("---- END get trans mat with translation -----")
        return trans_mat

    
    # def get_trans_mat_kinematic_chain(self, trans_mat_3_4):

    #     print("**** kinematic chain ****")
    #     trans_mat = trans_mat_3_4

    #     last_row = torch.tensor([0., 0., 0., 1.], device=trans_mat_3_4.device).repeat(*trans_mat_3_4.shape[:2], 1 , 1)
    #     trans_mat = torch.cat([trans_mat_3_4, last_row], dim=2)
    #     print("trans_mat", trans_mat.shape)
        
    #     # no root
    #     lev_1 = [0, 1, 2, 3, 4]
    #     lev_2 = [5, 6, 7, 8, 9]
    #     lev_3 = [10, 11, 12, 13, 14]
    #     lev_4 = [15, 16, 17, 18, 19]


    #     lev_1_mat = trans_mat[:, lev_1, : , :]
    #     lev_2_mat = torch.matmul(trans_mat[:, lev_2, : , :], lev_1_mat)
    #     lev_3_mat = torch.matmul(trans_mat[:, lev_3, : , :], lev_2_mat)
    #     lev_4_mat = torch.matmul(trans_mat[:, lev_4, : , :], lev_3_mat)

    #     print("lev_1_mat", lev_1_mat.shape)
    #     print("lev_4_mat", lev_4_mat.shape)

    #     final_mat = torch.cat([lev_1_mat, lev_2_mat, lev_3_mat, lev_4_mat], dim=1)
    #     # trans_mat = final_mat
    #     print("**** END kinematic chain ****")
    #     return trans_mat

    def from_3x3_mat_to_4x4(self, mat_3x3):
        last_col = torch.zeros(1, device=mat_3x3.device).repeat(*mat_3x3.shape[:2], 3, 1)
        mat_3x4 = torch.cat([mat_3x3, last_col], dim=3)
        # print("mat_3x3", mat_3x3.shape)
        last_row = torch.tensor([0., 0., 0., 1.], device=mat_3x4.device).repeat(*mat_3x4.shape[:2], 1 , 1)
        mat_4x4 = torch.cat([mat_3x4, last_row], dim=2)
        return mat_4x4

    def compute_adjusted_transpose(self, local_cs, rot_mat):
        lev_1 = [0, 1, 2, 3, 4]
        lev_2 = [5, 6, 7, 8, 9]
        lev_3 = [10, 11, 12, 13, 14]
        lev_4 = [15, 16, 17, 18, 19]

        lev_1_cs = local_cs[:, lev_1]
        lev_2_cs = local_cs[:, lev_2]
        lev_2_rot = rot_mat[:, lev_2]
        lev_3_cs = torch.matmul(lev_2_rot, local_cs[:, lev_3])
        lev_3_rot = torch.matmul(rot_mat[:, lev_3], lev_2_rot)
        lev_4_cs = torch.matmul(lev_3_rot, local_cs[:, lev_4])
        # lev_3_rot = torch.matmul(rot_mat[:, lev_3], lev_2_rot)

        adjust_cs = torch.cat([lev_1_cs, lev_2_cs, lev_3_cs, lev_4_cs], dim=1)

        loacl_cs_transpose = torch.transpose(local_cs, -2, -1) + 0
        loacl_cs_transpose[:, lev_3] = torch.matmul(loacl_cs_transpose[:, lev_3], lev_2_rot)

        loacl_cs_transpose[:, lev_4] = torch.matmul(loacl_cs_transpose[:, lev_4], lev_3_rot)

        transpose_cs = torch.transpose(adjust_cs, -2, -1)
        # transpose_cs = torch.inverse(adjust_cs) 
        return loacl_cs_transpose # transpose_cs

    def normalize_root_planes(self, bones, bone_lengths):
        # root = torch.zeros([3])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d', title='bones')
        # plot_local_coord(bones, bone_lengths, root, ax, show=False)
        bones_ori = bones + 0

        root_plane_norm_mat = torch.eye(3, device=self.dev).repeat(bones.shape[0], 20, 1, 1)
        # canonical_angle defines angles between root bone planes
        # angle between (n0,n1), (n1,n2), (n2,n3)
        canonical_angle = self.root_plane_angles
        # canonical_angle = np.array([0.8, 0.2, 0.2])
        bones_0 = bones[:, 0]
        bones_1 = bones[:, 1]
        bones_2 = bones[:, 2]
        bones_3 = bones[:, 3]
        bones_4 = bones[:, 4]

        # Use the plane between index(1) and middle(2) finger as reference plane (ref n1)
        # The normal of this plane is pointing toward y
        n1 = torch.cross(bones_2, bones_1)

        # Thumb and index (plane 0)
        n0 = torch.cross(bones_1, bones_0)
        n0_n1_angle = signed_angle(n0, n1, bones_1)
        # Rotate thumb root bone
        thumb_trans = rotation_matrix(n0_n1_angle - canonical_angle[0], bones_1)
        root_plane_norm_mat[:, 0] = thumb_trans
        # bones_plot = torch.matmul(root_plane_norm_mat, bones_ori.unsqueeze(-1)).squeeze(-1)
        # plot_local_coord(bones_plot, bone_lengths, root, ax, show=True)

        # Middle and ring (plane 2)
        n2 = torch.cross(bones_3, bones_2)
        n2_n1_angle = signed_angle(n2, n1, bones_2)
        # Rotate ring finger root bone, apply the same transformation to pinky
        ring_trans = rotation_matrix(n2_n1_angle + canonical_angle[1], bones_2)
        bones_3 = torch.matmul(ring_trans, bones_3.unsqueeze(-1)).squeeze(-1)
        bones_4 = torch.matmul(ring_trans, bones_4.unsqueeze(-1)).squeeze(-1)
        root_plane_norm_mat[:, 3] = ring_trans
        root_plane_norm_mat[:, 4] = ring_trans
        # bones_plot = torch.matmul(root_plane_norm_mat, bones_ori.unsqueeze(-1)).squeeze(-1)
        # plot_local_coord(bones_plot, bone_lengths, root, ax, show=True)

        # Ring and pinky (plane 3)
        n3 = torch.cross(bones_4, bones_3)
        n2 = torch.cross(bones_3, bones_2)
        n3_n2_angle = signed_angle(n3, n2, bones_3)
        # Rotate index finger root bone, apply the same transformation to thumb
        pinky_trans = rotation_matrix(n3_n2_angle + canonical_angle[2], bones_3)
        root_plane_norm_mat[:, 4] = torch.matmul(pinky_trans, ring_trans)
        # bones_plot = torch.matmul(root_plane_norm_mat, bones_ori.unsqueeze(-1)).squeeze(-1)
        # plot_local_coord(bones_plot, bone_lengths, root, ax, show=True)

        # Propagate rotations along kinematic chains
        for i in range(5):
            for j in range(3):
                root_plane_norm_mat[:, (j+1)*5 + i] = root_plane_norm_mat[:, i]
        
        new_bones = torch.matmul(root_plane_norm_mat.double(), bones_ori.unsqueeze(-1).double()).squeeze(-1)

        # plot_local_coord(new_bones, bone_lengths, root, ax)
        # import pdb; pdb.set_trace()

        return new_bones, root_plane_norm_mat

    def normalize_root_bone_angles(self, bones, bone_lengths):        
        # root = torch.zeros([3])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d', title='bones angle')
        # plot_local_coord(bones, bone_lengths, root, ax, show=False)

        bones_ori = bones + 0
        canonical_angle = 0.2 # 0.1
        # angle between (t,i), (i,m), (m,r), (r,p)
        # canonical_angle = self.root_bone_angles
        canonical_angle = np.array([0.4, 0.2, 0.2, 0.2])

        bones_0 = bones[:, 0]
        bones_1 = bones[:, 1]
        bones_2 = bones[:, 2]
        bones_3 = bones[:, 3]
        bones_4 = bones[:, 4]
        root_angle_norm_mat = torch.eye(3, device=self.dev).repeat(bones.shape[0], 20, 1, 1)
        # canonical_angle defines angles between adjacent bones
        # Use middle finger (f2) as reference
        # Plane normals always pointing out of the back of the hand

        # Index finger (f1), apply the same transformation to thumb (f0)
        n1 = cross(bones_2, bones_1, do_normalize=True)
        f2_f1_angle = signed_angle(bones_2, bones_1, n1)
        index_trans = rotation_matrix(canonical_angle[1] - f2_f1_angle, n1)
        root_angle_norm_mat[:, 1] =  index_trans
        root_angle_norm_mat[:, 0] =  index_trans
        bones_1 = torch.matmul(index_trans, bones_1.unsqueeze(-1)).squeeze(-1)
        bones_0 = torch.matmul(index_trans, bones_0.unsqueeze(-1)).squeeze(-1)
        # bones_plot = torch.matmul(root_angle_norm_mat, bones_ori.unsqueeze(-1)).squeeze(-1)
        # plot_local_coord(bones_plot, bone_lengths, root, ax, show=True)

        # Thumb (f0)
        n0 = cross(bones_1, bones_0, do_normalize=True)
        f1_f0_angle = signed_angle(bones_1, bones_0, n0)
        thumb_trans = rotation_matrix(canonical_angle[0] - f1_f0_angle, n0)
        root_angle_norm_mat[:, 0] = torch.matmul(thumb_trans, index_trans)
        bones_0 = torch.matmul(thumb_trans, bones_0.unsqueeze(-1)).squeeze(-1)
        # bones_plot = torch.matmul(root_angle_norm_mat, bones_ori.unsqueeze(-1)).squeeze(-1)
        # plot_local_coord(bones_plot, bone_lengths, root, ax, show=True)

        # Ring finger (f3), apply the same transformation to pinky finger (f4)
        # Notice the sign change in rotation_matrix()
        n2 = cross(bones_3, bones_2, do_normalize=True)
        f3_f2_angle = signed_angle(bones_3, bones_2, n2)
        ring_trans = rotation_matrix(f3_f2_angle - canonical_angle[2], n2)
        root_angle_norm_mat[:, 3] =  ring_trans
        root_angle_norm_mat[:, 4] =  ring_trans
        bones_3 = torch.matmul(ring_trans, bones_3.unsqueeze(-1)).squeeze(-1)
        bones_4 = torch.matmul(ring_trans, bones_4.unsqueeze(-1)).squeeze(-1)
        # bones_plot = torch.matmul(root_angle_norm_mat, bones_ori.unsqueeze(-1)).squeeze(-1)
        # plot_local_coord(bones_plot, bone_lengths, root, ax, show=True)

        # Pinky finger (f4)
        n3 = cross(bones_4, bones_3, do_normalize=True)
        f4_f3_angle = signed_angle(bones_4, bones_3, n3)
        pinky_trans = rotation_matrix(f4_f3_angle - canonical_angle[3], n3)
        root_angle_norm_mat[:, 4] = torch.matmul(pinky_trans, ring_trans)
        bones_4 = torch.matmul(pinky_trans, bones_4.unsqueeze(-1)).squeeze(-1)

        # bones_plot = torch.matmul(root_angle_norm_mat, bones_ori.unsqueeze(-1)).squeeze(-1)
        # plot_local_coord(bones_plot, bone_lengths, root, ax, show=True)

        # Propagate rotations along kinematic chains
        for i in range(5):
            for j in range(3):
                root_angle_norm_mat[:, (j+1)*5 + i] = root_angle_norm_mat[:, i]
        
        new_bones = torch.matmul(root_angle_norm_mat.double(), bones_ori.unsqueeze(-1).double()).squeeze(-1)

        # plot_local_coord(new_bones, bone_lengths, root, ax, show=True)
        # import pdb; pdb.set_trace()
        
        return new_bones, root_angle_norm_mat

    def forward(self, joints, kp3d_is_right, return_rot_only=False):

        assert joints.size(1) == 21, "Number of joints needs to be 21"

        nrb_idx = self.nrb_idx
        # Pre-process the joints
        joints = self.preprocess_joints(joints, kp3d_is_right)
        # Compute the bone vectors
        bones, bone_lengths, kp_to_bone_mat = self.kp3D_to_bones(joints)
        bone_tmp = bones + 0

        # New normalization
        # Normalize the root bone planes
        plane_normalized_bones, root_plane_norm_mat = self.normalize_root_planes(bones, bone_lengths)
        # Normalize angles between root bones
        angle_normalized_bones, root_angle_norm_mat = self.normalize_root_bone_angles(plane_normalized_bones, bone_lengths)
        bones = angle_normalized_bones

        # Plot root bone normalization
        # root = torch.zeros([3])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d', title='bones angle')
        # plot_local_coord(bones, bone_lengths, root, ax, show=False)
        # plot_local_coord(angle_normalized_bones, bone_lengths, root, ax, show=True)

        # Combine plane normalization and angle normalization
        root_bones_norm_mat = torch.matmul(root_angle_norm_mat, root_plane_norm_mat) # root_plane_norm_mat
        # print("root_bones_norm_mat", root_bones_norm_mat.shape)
        # root_bones_norm_mat = torch.eye(3, device=bones.device).reshape(1, 1, 3, 3).repeat(*bones.shape[0:2], 1, 1)
        # print("root_bones_norm_mat", root_bones_norm_mat.shape)

        # Compute the local coordinate systems for each bone
        # This assume the root bones are fixed
        local_cs = self.compute_local_coordinate_system(bones.double())
        # Compute the local coordinates
        local_coords = self.compute_local_coordinates(bones.float(), local_cs.float())

        # root = torch.zeros([1,21,3])
        root = torch.zeros([3])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d', title='local_coords')
        # # plot_local_coord_system(local_cs, bone_lengths, root, ax)
        # plot_local_coord(local_coords, bone_lengths, root, ax)

        # Copmute the rotation around the y and rotated-x axis
        rot_angles = self.compute_rot_angles(local_coords)
        if return_rot_only:
            nrb_rot_angles = rot_angles[:, nrb_idx]
            return nrb_rot_angles

        # print("rot angles", rot_angles)
        # Compute rotation matrix
        rot_mat = self.compute_rotation_matrix(rot_angles.float(), local_coords.float())
        # print("rot mat", rot_mat.shape)
        # print("local_cs", local_cs.shape)

        # loacl_cs_transpose = torch.transpose(local_cs, -2, -1)
        loacl_cs_transpose = self.compute_adjusted_transpose(local_cs, rot_mat)
        # print("loacl_cs_transpose", loacl_cs_transpose.shape)
        # should_be_i = torch.matmul(loacl_cs_transpose, local_cs)
        # print("sanity check", should_be_i)
        # print("sanity check transpose", torch.bmm(loacl_cs_transpose, local_cs))
        trans_mat_without_scale_translation = torch.matmul(loacl_cs_transpose, torch.matmul(rot_mat, local_cs))
        # print("trans_mat_without_scale_translation", trans_mat_without_scale_translation)

        ####
        # local_coords_no_back_proj = self.compute_local_coordinates(bones, torch.matmul(rot_mat, local_cs))
        # print("--- local_coords_no_back_proj ---", local_coords_no_back_proj)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plot_local_coord(local_coords_no_back_proj, bone_lengths, root, ax)

        # Compute local coordinates of each bone after unposing to adjust keypoint translation
        local_coords_after_unpose = self.compute_local_coordinates(bones.float(), trans_mat_without_scale_translation.float())
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # # plot_local_coord(local_coords_no_back_proj, bone_lengths, root, ax)
        # plot_local_coord(local_coords_after_unpose, bone_lengths, root, ax) # , show=False)

        #### normal transpose
        # loacl_cs_normal_transpose = torch.transpose(local_cs, -2, -1)
        # trans_mat_normal_transpose = torch.matmul(loacl_cs_normal_transpose, torch.matmul(rot_mat, local_cs))
        # # print("loacl_cs_transpose", loacl_cs_transpose.shape)
        # local_coords_normal_transpose = self.compute_local_coordinates(bones, trans_mat_normal_transpose)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plot_local_coord(local_coords_no_back_proj, bone_lengths, root, ax)
        # print("normal transpose")
        # plot_local_coord(local_coords_normal_transpose, bone_lengths, root, ax)


        # print("bone_lengths", bone_lengths, bone_lengths.shape)

        # scale_mat = self.get_scale_mat_from_bone_lengths(bone_lengths)
        # print("scale_mat", scale_mat, scale_mat.shape)
        # trans_mat_without_translation = torch.matmul(scale_mat, trans_mat_without_scale_translation)

        # This return 3 x 4 transformation matrix with translation
        # trans_mat_with_translation = self.get_trans_mat_with_translation(
        #     trans_mat_without_scale_translation, local_coords_after_unpose, bones, bone_lengths)

        # This return 4 x 4 transformation matrix
        # trans_mat_kinematic_chain = self.get_trans_mat_kinematic_chain(trans_mat_with_translation)
        
        trans_mat_without_scale_translation = self.from_3x3_mat_to_4x4(trans_mat_without_scale_translation)

        # Convert bones back to keypoints
        inv_scale_trans = self.compute_bone_to_kp_mat(bone_lengths, local_coords_after_unpose)
        # Combine everything into one transformation matrix from posed keypoints to unposed keypoints
        # import pdb; pdb.set_trace()
        # root_bones_norm_mat
        trans_mat = torch.matmul(self.from_3x3_mat_to_4x4(root_bones_norm_mat.float()), kp_to_bone_mat.float())
        trans_mat = torch.matmul(trans_mat_without_scale_translation, trans_mat)
        # trans_mat = torch.matmul(trans_mat_without_scale_translation, kp_to_bone_mat)
        trans_mat = torch.matmul(inv_scale_trans.double(), trans_mat.double())
        # Add root keypoint tranformation
        # import pdb; pdb.set_trace()
        root_trans = torch.eye(4, device=trans_mat.device).reshape(1, 1, 4, 4).repeat(trans_mat.shape[0], 1, 1, 1)
        trans_mat = torch.cat([root_trans, trans_mat], dim=1)

        bone_lengths = torch.cat([torch.ones([trans_mat.shape[0], 1, 1], device=trans_mat.device), bone_lengths], dim=1)


        # Compute the angle loss
        # def interval_loss(x, min_v, max_v):
        #     if min_v.dim() == 1:
        #         min_v = min_v.unsqueeze(0)
        #         max_v = max_v.unsqueeze(0)
        #     zero = self.zero
        #     return (torch.max(min_v - x, zero) + torch.max(x - max_v, zero))
        # Discard the root bones
        nrb_rot_angles = rot_angles[:, nrb_idx]
        # Compute the polygon distance
        # poly_d = self.polygon_distance(nrb_rot_angles)
        # Compute the final loss
        # per_batch_loss = poly_d.mean(1)
        # angle_loss = per_batch_loss.mean()
        
        # Storage for debug purposes
        # self.per_batch_loss = per_batch_loss.detach()
        self.bones = bones.detach()
        self.local_cs = local_cs.detach()
        self.local_coords = local_coords.detach()
        self.nrb_rot_angles = nrb_rot_angles.detach()
        # self.loss_per_sample = poly_d.detach()

        return trans_mat, bone_lengths # rot_mat # rot_angles # angle_loss


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    import matplotlib.pyplot as plt
    plt.ion()
    from pose.utils.visualization_2 import plot_fingers
    import yaml
    from tqdm import tqdm
    from prototypes.utils import get_data_reader

    dev = torch.device('cpu')
    # Load constraints
    cfg_path = "hp_params/all_params.yaml"
    hand_constraints = yaml.load(open(cfg_path).read())
    # Hand parameters
    for k,v in hand_constraints.items():
        if isinstance(v, list):
            hand_constraints[k] = torch.from_numpy(np.array(v)).float()

    angle_poly = hand_constraints['convex_hull']
    angle_loss = AngleLoss(angle_poly, dev=dev)
    ##### Consistency test. Make sure loss is 0 for all samples
    # WARNING: This will fail because we approximate the angle polygon
    # Get data reader
    data_reader = get_data_reader(ds_name='stb', is_train=True)
    # Make sure error is 0 for all samples of the training set
    tol = 1e-8
    for i in tqdm(range(len(data_reader))):
        sample = data_reader[i]
        kp3d = sample["joints3d"].view(-1,21,3)
        is_right = sample["kp3d_is_right"].view(-1,1)
        loss = angle_loss(kp3d, is_right)
        import pdb;pdb.set_trace()
        if loss > tol:
            print("ERROR")
            plot_fingers(kp3d[0])
            import pdb;pdb.set_trace()
        # Shifting shouldnt cause an issue neither
        kp3d_center = kp3d - kp3d[:,0:1]
        loss = angle_loss(kp3d_center, is_right)
        if loss > tol:
            print("ERROR")
            plot_fingers(kp3d_center[0])
            import pdb;pdb.set_trace()
        # Scaling should be 0 error too
        kp3d_scale = kp3d * 10
        loss = angle_loss(kp3d_scale, is_right)
        if loss > tol:
            print("ERROR")
            plot_fingers(kp3d_scale[0])
            import pdb;pdb.set_trace()

    ##### SGD Test
    # torch.manual_seed(4)
    # x = torch.zeros(1,21,3)
    # x[:,1:6] = torch.rand(1,5,3)
    # is_right = torch.tensor(1.0).view(1,1)
    # x.requires_grad_()
    # print_freq = 100
    # lr = 1e-1
    # ax = None

    # i = 0
    # while True:
    # # while i < 100:
        # loss = root_bone_loss(x, is_right)
        # loss.backward()
        # if (i % print_freq) == 0:
            # print("It: %d\tLoss: %.08f" % (i,loss.item()))
            # to_plot = x[0].clone()
            # ax = plot_fingers(to_plot, ax=ax, set_view=False)
            # to_plot = to_plot.detach().numpy()
            # ax.plot(to_plot[1:6,0], to_plot[1:6,1], to_plot[1:6,2], 'b')
            # plt.show()
            # plt.pause(0.001)
            # if i == 0:
                # # Pause for initial conditions
                # input()

        # with torch.no_grad():
            # x = x - lr * x.grad

        # x.requires_grad_()
        # i += 1

