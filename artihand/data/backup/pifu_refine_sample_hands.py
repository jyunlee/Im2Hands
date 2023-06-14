import os
import open3d as o3d
import sys
from glob import glob
import logging
import torch
import torchvision.transforms
from torch.utils import data
import numpy as np
import yaml
import cv2 as cv
import pickle
import json
from manopth.manolayer import ManoLayer

sys.path.insert(0, "/workspace/halo/halo")

from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import (convert_joints, change_axes, scale_halo_trans_mat)
from models.halo_adapter.projection import get_projection_layer
from models.halo_adapter.transform_utils import xyz_to_xyz1

sys.path.append('/workspace/IntagHand')

from dataset.dataset_utils import IMG_SIZE, HAND_BBOX_RATIO, HEATMAP_SIGMA, HEATMAP_SIZE, cut_img

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


def preprocess_joints(left_joints, right_joints):

    permute_mat = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]

    left_joints = left_joints - left_joints[0]
    #left_joints /= 1000
    left_joints = left_joints[permute_mat]
    left_joints *= [-1, 1, 1]

    right_joints = right_joints - right_joints[0]
    #right_joints /= 1000
    right_joints = right_joints[permute_mat]

    for side in ['left', 'right']:

        ''' KPTS DATA PRE-PROCESSING '''
        if side == 'left':
            joints = left_joints
        else:
            joints = right_joints

        hand_joints = torch.Tensor(joints).unsqueeze(0)
        joints = convert_joints(hand_joints, source='halo', target='biomech')
        is_right_vec = torch.ones(joints.shape[0], device='cpu')

        pose_converter = PoseConverter(dev='cpu')

        palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(joints.double(), is_right=is_right_vec)

        palm_align_kps_local_cs_nasa_axes, swap_axes_mat = change_axes(palm_align_kps_local_cs)

        swap_axes_mat = swap_axes_mat.unsqueeze(0)
        rot_then_swap_mat = torch.matmul(swap_axes_mat.float(), glo_rot_right.float()).unsqueeze(0)

        trans_mat_pc, _ = pose_converter(palm_align_kps_local_cs_nasa_axes.double(), is_right_vec.double())
        trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='halo')

        joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])
        trans_mat_pc = trans_mat_pc[:, joints_for_nasa_input]

        hand_joints = hand_joints[0]
        org_hand_joints = hand_joints.cpu().numpy()

        hand_joints = torch.matmul(rot_then_swap_mat.squeeze().cpu(), xyz_to_xyz1(hand_joints.cpu()).unsqueeze(-1))[:, :3, 0]
        bone_lengths = get_bone_lengths(hand_joints)

        trans_mat_pc_all = trans_mat_pc
        unpose_mat = scale_halo_trans_mat(trans_mat_pc_all)

        scale = 0.4

        scale_mat = np.identity(4) * scale
        scale_mat[3,3] = 1.
        unpose_mat = np.matmul(unpose_mat.cpu().numpy(), scale_mat)

        unpose_mat = torch.Tensor(unpose_mat)

        if side == 'left':
            left_inputs = unpose_mat.squeeze()
            left_bone_lengths = torch.Tensor(bone_lengths).squeeze(0)
            left_root_rot_mat = rot_then_swap_mat.squeeze()

        else:
            right_inputs = unpose_mat.squeeze()  
            right_bone_lengths = torch.Tensor(bone_lengths).squeeze()
            right_root_rot_mat = rot_then_swap_mat.squeeze()
    
    return left_inputs, left_bone_lengths, left_root_rot_mat, right_inputs, right_bone_lengths, right_root_rot_mat, left_joints, right_joints


class PIFuRefineSampleHandDataset(data.Dataset):

    def __init__(self, data_path, mano_dataset_folder, input_helpers, split=None,
            no_except=True, transforms=None, return_idx=False, subset=1, split_idx=0, lower_bound=0):

        assert split in ['train', 'test', 'val']

        self.split = split
        self.subset = subset
        self.split_idx = split_idx
        self.lower_bound = lower_bound

        self.mano_dataset_folder = mano_dataset_folder
        self.input_helpers = input_helpers
        self.no_except = no_except
        self.transforms = transforms
        self.return_idx = return_idx

        self.data_path = data_path
        self.size = len(glob(os.path.join(data_path, split, 'anno', '*.pkl')))

        self.normalize_img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with open(os.path.join(mano_dataset_folder, split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
            self.annot_file = json.load(f)

        self.right_mano_layer = ManoLayer(
                mano_root='/workspace/mano_v1_2/models', use_pca=False, ncomps=45, flat_hand_mean=False)
        self.left_mano_layer = ManoLayer(
                mano_root='/workspace/mano_v1_2/models', use_pca=False, ncomps=45, flat_hand_mean=False, side='left')

    def __len__(self):
        return (self.size - self.lower_bound) // self.subset

    def __getitem__(self, idx):
        idx *= self.subset
        idx += self.lower_bound
        idx += self.split_idx

        print(os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))

        img = cv.imread(os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        hms = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}.jpg'.format(idx)))
        mask = cv.imread(os.path.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        dense = cv.imread(os.path.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        mask = cv.resize(mask, (IMG_SIZE, IMG_SIZE))
        maskTensor = torch.tensor(cv.cvtColor(mask, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        maskTensor = maskTensor.permute(2, 0, 1)
        maskTensor = self.normalize_img(maskTensor)

        dense = cv.resize(dense, (IMG_SIZE, IMG_SIZE))
        denseTensor = torch.tensor(cv.cvtColor(dense, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        denseTensor = denseTensor.permute(2, 0, 1)
        denseTensor = self.normalize_img(denseTensor)

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        camera_params = {}

        camera_params['R'] = data['camera']['R']
        camera_params['T'] = data['camera']['t']
        camera_params['camera'] = data['camera']['camera']

        capture_idx = data['image']['capture']
        frame_idx = data['image']['frame_idx']
        seq_name = data['image']['seq_name']

        split_path = os.path.join(self.mano_dataset_folder, self.split)

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
                            # % (self.input_helper.__class__.__name__, model)
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

        if self.transforms is not None:
            for side in ['right', 'left']:
                for tran_name, tran in self.transforms.items():
                    mano_data[side] = tran(mano_data[side])
                # if field_name in self.transform:
                #     data[field_name] = self.transform[field_name](data[field_name])

        if self.return_idx:
            mano_data[side]['idx'] = idx

        camera_params['right_root_xyz'] = mano_data['right']['root_xyz']
        camera_params['left_root_xyz'] = mano_data['left']['root_xyz']

        print(idx)
        # this part is added
        pred_left_joints = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_path, self.split, 'pred_joints', '%07d_left.ply' % idx)).points)
        pred_right_joints = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_path, self.split, 'pred_joints', '%07d_right.ply' % idx)).points)

        left_inputs, left_bone_lengths, left_root_rot_mat, right_inputs, right_bone_lengths, right_root_rot_mat, _, _  = preprocess_joints(pred_left_joints, pred_right_joints)

        '''
        print(mano_data['left']['inputs'])
        print(left_inputs)

        print()
        
        print(mano_data['left']['bone_lengths'])
        print(left_bone_lengths)

        print()

        print(mano_data['left']['root_rot_mat'])
        print(left_root_rot_mat)
        exit()
        '''
        mano_data['left']['inputs'] = left_inputs
        mano_data['left']['bone_lengths'] = left_bone_lengths
        mano_data['left']['root_rot_mat'] = left_root_rot_mat

        mano_data['right']['inputs'] = right_inputs
        mano_data['right']['bone_lengths'] = right_bone_lengths
        mano_data['right']['root_rot_mat'] = right_root_rot_mat

        return imgTensor, maskTensor, denseTensor, camera_params, mano_data, idx

