# For interfacing with the HALO mesh model code
import argparse
import trimesh
import numpy as np
import os
import torch
import sys

sys.path.insert(0, "../../halo_base")
#from artihand import config #, data
from artihand.checkpoints import CheckpointIO


def get_halo_model(config_file):
    '''
    Args:
        config_file (str): HALO config file
    '''
    no_cuda = False
    print("config_file", config_file)
    cfg = config.load_config(config_file, '../halo_base/configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # Model
    model = config.get_model(cfg, device=device)

    out_dir = cfg['training']['out_dir']
    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # print(checkpoint_io.module_dict['model'])

    # print(model.state_dict().keys())

    # Generator
    generator = config.get_generator(model, cfg, device=device)
    # print("upsampling", generator.upsampling_steps)
    return model, generator


def convert_joints(joints, source, target):
    halo_joint_to_mano = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
    mano_joint_to_halo = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20])
    mano_joint_to_biomech = np.array([0, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20])
    biomech_joint_to_mano = np.array([0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20])
    halo_joint_to_biomech = np.array([0, 13, 1, 4, 10, 7, 14, 2, 5, 11, 8, 15, 3, 6, 12, 9, 16, 17, 18, 19, 20])
    # halo_joint_to_biomech = np.array([0, 11, 15, 19, 4, 1, 5, 9, 8, 13, 17, 2, 12, 18, 3, 7, 16, 6, 10, 14, 20])
    biomech_joint_to_halo = np.array([0, 2, 7, 12, 3, 8, 13, 5, 10, 15, 4, 9, 14, 1, 6, 11, 16, 17, 18, 19, 20])

    if source == 'halo' and target == 'biomech':
        # conn = nasa_joint_to_mano[mano_joint_to_biomech]
        # print(conn)
        # tmp_mano = joints[:, halo_joint_to_mano]
        # out = tmp_mano[:, mano_joint_to_biomech]
        return joints[:, halo_joint_to_biomech]
    if source == 'biomech' and target == 'halo':
        return joints[:, biomech_joint_to_halo]
    if source == 'mano' and target == 'biomech':
        return joints[:, mano_joint_to_biomech]
    if source == 'biomech' and target == 'mano':
        return joints[:, biomech_joint_to_mano]
    if source == 'halo' and target == 'mano':
        return joints[:, halo_joint_to_mano]
    if source == 'mano' and target == 'halo':
        return joints[:, mano_joint_to_halo]

    print("-- Undefined convertion. Return original tensor --")
    return joints


def change_axes(keypoints, source='mano', target='halo'):
    """Swap axes to match that of NASA
    """
    # Swap axes from local_cs to NASA
    kps_halo = keypoints + 0
    kps_halo[..., 0] = keypoints[..., 1]
    kps_halo[..., 1] = keypoints[..., 2]
    kps_halo[..., 2] = keypoints[..., 0]

    mat = torch.zeros([4, 4], device=keypoints.device)
    mat[0, 1] = 1.
    mat[1, 2] = 1.
    mat[2, 0] = 1.
    mat[3, 3] = 1.

    return kps_halo, mat


def get_bone_lengths(joints, source='biomech', target='halo'):
    ''' To get the bone lengths for halo inputs
    '''
    joints = convert_joints(joints, source=source, target=target)
    bones_idx = np.array([
        (0, 4),  # use distance from root to middle finger as palm bone length
        (1, 2),
        (2, 3),
        (3, 17),
        (4, 5),
        (5, 6),
        (6, 18),
        (7, 8),
        (8, 9),
        (9, 20),
        (10, 11),
        (11, 12),
        (12, 19),
        (13, 14),
        (14, 15),
        (15, 16)
    ])
    bones = joints[:, bones_idx[:, 0]] - joints[:, bones_idx[:, 1]]
    bone_lengths = torch.norm(bones, dim=-1)
    return bone_lengths


def scale_halo_trans_mat(trans_mat, scale=0.4):
    ''' Scale the transformation matrices to match the scale of HALO.
        Maybe this should be in the HALO package.
    Args:
        trans_mat: Transformation matrices that are already inverted (from pose to unpose)
    '''
    # Transform meta data
    # Assume that the transformation matrices are already inverted
    scale_mat = torch.eye(4, device=trans_mat.device).reshape(1, 1, 4, 4).repeat(trans_mat.shape[0], 1, 1, 1) * scale
    scale_mat[:, :, 3, 3] = 1.

    nasa_input = torch.matmul(trans_mat.double(), scale_mat.double())
    # (optional) scale canonical pose by the same global scale to make learning occupancy function easier
    canonical_scale_mat = torch.eye(4, device=trans_mat.device).reshape(1, 1, 4, 4).repeat(trans_mat.shape[0], 1, 1, 1) / scale
    canonical_scale_mat[:, :, 3, 3] = 1.
    nasa_input = torch.matmul(canonical_scale_mat.double(), nasa_input.double())
    return nasa_input
