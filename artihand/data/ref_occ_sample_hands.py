import os
import sys
import json
import pickle
import logging
import trimesh
import torch
import torchvision.transforms

import numpy as np
import cv2 as cv
import open3d as o3d

from glob import glob
from torch.utils import data
from manopth.manolayer import ManoLayer
from dependencies.halo.halo_adapter.converter import PoseConverter, transform_to_canonical
from dependencies.halo.halo_adapter.interface import (get_halo_model, convert_joints, change_axes, scale_halo_trans_mat)
from dependencies.halo.halo_adapter.projection import get_projection_layer
from dependencies.halo.halo_adapter.transform_utils import xyz_to_xyz1
from dependencies.intaghand.dataset.dataset_utils import IMG_SIZE, HAND_BBOX_RATIO, HEATMAP_SIGMA, HEATMAP_SIZE, cut_img

logger = logging.getLogger(__name__)


def get_bone_lengths(joints):
    bones = np.array([
        (0,4),
        (1,2),
        (2,3),
        (3,17),
        (4,5),
        (5,6),
        (6,18),
        (7,8),
        (8,9),
        (9,20),
        (10,11),
        (11,12),
        (12,19),
        (13,14),
        (14,15),
        (15,16)
    ])

    bone_length = joints[bones[:,0]] - joints[bones[:,1]]
    bone_length = np.linalg.norm(bone_length, axis=1)

    return bone_length


# Codes adopted from HALO
def preprocess_joints(joints, side='right', scale=0.4):

    permute_mat = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]

    joints -= joints[0]
    joints = joints[permute_mat]

    if side == 'left':
        joints *= [-1, 1, 1]

    org_joints = joints

    joints = torch.Tensor(joints).unsqueeze(0)
    joints = convert_joints(joints, source='halo', target='biomech')
    is_right_vec = torch.ones(joints.shape[0])

    pose_converter = PoseConverter()

    palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(joints.double(), is_right=is_right_vec)
    palm_align_kps_local_cs_nasa_axes, swap_axes_mat = change_axes(palm_align_kps_local_cs)

    rot_then_swap_mat = torch.matmul(swap_axes_mat.unsqueeze(0), glo_rot_right.float()).unsqueeze(0)

    trans_mat_pc, _ = pose_converter(palm_align_kps_local_cs_nasa_axes, is_right_vec)
    trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='halo')

    joints_for_nasa_input = [0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    trans_mat_pc = trans_mat_pc[:, joints_for_nasa_input]

    org_joints = torch.matmul(rot_then_swap_mat.squeeze(), xyz_to_xyz1(torch.Tensor(org_joints)).unsqueeze(-1))[:, :3, 0]
    bone_lengths = torch.Tensor(get_bone_lengths(org_joints)).squeeze()

    trans_mat_pc_all = trans_mat_pc
    unpose_mat = scale_halo_trans_mat(trans_mat_pc_all)

    scale_mat = torch.eye(4) * scale
    scale_mat[3, 3] = 1.
    unpose_mat = torch.matmul(unpose_mat, scale_mat.double()).squeeze()

    return unpose_mat, torch.Tensor(bone_lengths).squeeze(0), rot_then_swap_mat.squeeze()


class RefOccSampleHandDataset(data.Dataset):

    def __init__(self, data_path, anno_path, input_helpers, split=None,
                 no_except=True, transforms=None, subset=1, subset_idx=0):

        assert split in ['train', 'test', 'val']

        self.split = split
        self.subset = subset
        self.subset_idx = subset_idx

        self.input_helpers = input_helpers
        self.no_except = no_except
        self.transforms = transforms

        self.data_path = data_path
        self.anno_path = anno_path

        self.size = len(glob(os.path.join(data_path, split, 'anno', '*.pkl')))

        self.normalize_img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with open(os.path.join(anno_path, split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
            self.annot_file = json.load(f)

        self.right_mano_layer = ManoLayer(
                mano_root='/workspace/mano_v1_2/models', use_pca=False, ncomps=45, flat_hand_mean=False)
        self.left_mano_layer = ManoLayer(
                mano_root='/workspace/mano_v1_2/models', use_pca=False, ncomps=45, flat_hand_mean=False, side='left')


    def __len__(self):
        return self.size // self.subset


    def __getitem__(self, idx):
        idx = idx * self.subset + self.subset_idx
        
        img_path = os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx))

        img = cv.imread(os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))

        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        pred_left_joints = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_path, self.split, 'pred_joints', '%07d_left.ply' % idx)).points)
        pred_right_joints = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_path, self.split, 'pred_joints', '%07d_right.ply' % idx)).points)

        pred_left_shape = trimesh.load(os.path.join(self.data_path, self.split, 'pred_shapes', '%07d_left.obj' % idx))
        pred_right_shape = trimesh.load(os.path.join(self.data_path, self.split, 'pred_shapes', '%07d_right.obj' % idx))

        camera_params = {}
        camera_params['R'] = data['camera']['R']
        camera_params['T'] = data['camera']['t']
        camera_params['camera'] = data['camera']['camera']

        capture_idx = data['image']['capture']
        frame_idx = data['image']['frame_idx']
        seq_name = data['image']['seq_name']

        split_path = os.path.join(self.anno_path, self.split)

        mano_data = {'right': {}, 'left': {}}

        for side in ['right', 'left']:
            for field_name, input_helper in self.input_helpers.items():
                try:
                    model = '%s_%s_%s' % (capture_idx, frame_idx, side) # !!! TO BE MODIFIED
                    field_data = input_helper.load(split_path, model)
                except Exception:
                    if self.no_except:
                        logger.warn(
                            'Error occured when loading field %s of model %s'
                            % (field_name.__class__.__name__, model)
                        )
                        return None
                    else:
                        raise

                if isinstance(field_data, dict):
                    for k, v in field_data.items():
                        if k is None:
                            mano_data[side][field_name] = v
                        elif field_name == 'inputs':
                            mano_data[side][k] = v
                        else:
                            mano_data[side]['%s.%s' % (field_name, k)] = v
                else:
                    mano_data[side][field_name] = field_data

        camera_params['right_root_xyz'] = mano_data['right']['root_xyz']
        camera_params['left_root_xyz'] = mano_data['left']['root_xyz']

        if self.transforms is not None:
            for side in ['right', 'left']:
                for tran_name, tran in self.transforms.items():
                    mano_data[side] = tran(mano_data[side])

        mano_data[side]['idx'] = idx

        left_inputs, left_bone_lengths, left_root_rot_mat = preprocess_joints(pred_left_joints, side='left')
        right_inputs, right_bone_lengths, right_root_rot_mat = preprocess_joints(pred_right_joints)

        left_anchor_points = trimesh.sample.sample_surface_even(pred_left_shape, 512)[0]  # sample_surface_even
        right_anchor_points = trimesh.sample.sample_surface_even(pred_right_shape, 512)[0]

        if left_anchor_points.shape[0] != 512:
            left_anchor_points = np.concatenate((left_anchor_points, left_anchor_points[:512-left_anchor_points.shape[0]]), 0)
        if right_anchor_points.shape[0] != 512:
            right_anchor_points = np.concatenate((right_anchor_points, right_anchor_points[:512-right_anchor_points.shape[0]]), 0)

        mano_data['left']['pred_joints'] = pred_left_joints
        mano_data['left']['inputs'] = left_inputs
        mano_data['left']['bone_lengths'] = left_bone_lengths
        mano_data['left']['root_rot_mat'] = left_root_rot_mat
        mano_data['left']['mid_joint'] = (pred_left_joints - pred_left_joints[0])[9]
        mano_data['left']['anchor_points'] = left_anchor_points

        mano_data['right']['pred_joints'] = pred_right_joints
        mano_data['right']['inputs'] = right_inputs
        mano_data['right']['bone_lengths'] = right_bone_lengths
        mano_data['right']['root_rot_mat'] = right_root_rot_mat
        mano_data['right']['mid_joint'] = (pred_right_joints - pred_right_joints[0])[9]
        mano_data['right']['anchor_points'] = right_anchor_points

        camera_params['img_path'] = img_path

        return imgTensor, camera_params, mano_data, idx

