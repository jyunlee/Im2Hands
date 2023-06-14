import os
import logging
from matplotlib.pyplot import axis
from torch.utils import data
import numpy as np
import yaml
import pickle
import torch
from scipy.spatial import distance

from models.data.input_helpers import random_rotate
from models.utils import visualize as vis
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


logger = logging.getLogger(__name__)

class ObmanDataset(data.Dataset):
    ''' Obman dataset class.
    '''

    def __init__(self, dataset_folder, input_helpers=None, split=None,
                 no_except=True, transforms=None, return_idx=False, use_bps=False, random_rotate=False):
        ''' Initialization of the the 3D articulated hand dataset.
        Args:
            dataset_folder (str): dataset folder
            input_helpers dict[(callable)]: helpers for data loading
            split (str): which split is used
            no_except (bool): no exception
            transform dict{(callable)}: transformation applied to data points
            return_idx (bool): wether to return index
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.input_helpers = input_helpers
        self.split = split
        self.no_except = no_except
        self.transforms = transforms
        self.return_idx = return_idx
        self.use_bps = use_bps

        self.obman_data = False

        if self.use_bps:
            split_file = os.path.join(dataset_folder, split + '_bps.npz')
            all_data = np.load(split_file)
            self.object_bps = all_data["object_bps"]
        elif self.obman_data:
            split_file = os.path.join(dataset_folder, split + '.pkl')
            with open(split_file, "rb") as data_file:
                all_data = pickle.load(data_file)
            self.hand_joints = all_data["hand_joints3d"]
            self.object_points = all_data["object_points3d"]

        else:
            split_file = os.path.join(dataset_folder, split + '_hand.npz')
            all_data = np.load(split_file)

            self.hand_joints = all_data["hand_joints3d"]
            self.object_points = all_data["object_points3d"]
            self.closest_point_idx = all_data["closest_obj_point_idx"]
            self.closest_point_dist = all_data["closest_obj_point_dist"]
            self.obj_names = all_data["obj_name"]
            self.rot_mats = all_data["rot_mat"]

            # load sample point inside
            sample_point_file = os.path.join(dataset_folder, split + '_sample_vol.npz')
            sample_points = np.load(sample_point_file)
            points_dict = {}
            for k in sample_points.files:
                points_dict[k] = sample_points[k]

            self.sample_points = points_dict
            self.hand_verts = all_data["hand_verts"]

        if split == 'val':
            val_keep = 1500
            if len(self.hand_joints) > val_keep:
                keep_idx = np.random.choice(len(self.hand_joints), val_keep, replace=False)
                if self.obman_data:
                    keep_hand_joints = []
                    keep_object_points = []
                    for idx in keep_idx:
                        keep_hand_joints.append(self.hand_joints[idx])
                        keep_object_points.append(self.object_points[idx])
                    self.hand_joints = keep_hand_joints
                    self.object_points = keep_object_points
                else:
                    self.hand_joints = self.hand_joints[keep_idx]
                    self.object_points = self.object_points[keep_idx]
                    self.closest_point_idx = self.closest_point_idx[keep_idx]
                    self.closest_point_dist = self.closest_point_dist[keep_idx]
                    self.hand_verts = self.hand_verts[keep_idx]
                    if self.use_bps:
                        self.object_bps = self.object_bps[keep_idx]                

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.hand_joints)  # len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        data = {}
        data['mesh_path'] = ''
        data['object_points'] = self.object_points[idx]
        data['hand_joints'] = self.hand_joints[idx]
        data['object_points'], data['hand_joints'], data['obj_center'] = self.obj_to_origin(data['object_points'], data['hand_joints'])

        if not self.obman_data:
            data['closest_point_idx'] = self.closest_point_idx[idx]
            data['closest_point_dist'] = self.closest_point_dist[idx]
            # hand verts
            data['hand_verts'] = self.hand_verts[idx] - data['obj_center']

            # Get points sampled inside the mesh for occupancy query
            inside_points = self.sample_points[self.obj_names[idx]]
            keep_idx = np.random.choice(len(inside_points), 600, replace=False)
            inside_points = inside_points[keep_idx]
            # inside_points = np.inside_points * self.rot_mats[idx].T
            inside_points = np.matmul(self.rot_mats[idx], np.expand_dims(inside_points, -1)).squeeze(-1)
            data['inside_points'] = inside_points - data['obj_center']

        if self.use_bps:
            data['object_bps'] = self.object_bps[idx]
        else:
            data = self.scale_to_cm(data)
        data = self.gen_refine_training_data(data)

        if self.return_idx:
            data['idx'] = idx

        return data

    def gen_refine_training_data(self, data):
        hand_joints, obj_points = data['hand_joints'], data['object_points']
        mu = 0.0
        scale = 0.5  # 2mm
        noise = np.random.normal(mu, scale, 15 * 3)
        noise = noise.reshape((15, 3))
        noisy_joints = hand_joints.copy()
        noisy_joints[6:] = noisy_joints[6:] + noise

        trans_noise = np.random.rand() * 2.0  # 0.5
        noisy_joints = noisy_joints + trans_noise

        data['noisy_joints'] = noisy_joints
        tip_idx = np.array([4, 8, 12, 16, 20])
        p2p_dist = distance.cdist(noisy_joints, obj_points)
        p2p_dist = p2p_dist.min(axis=1)
        data['tip_dists'] = p2p_dist
        return data

    def obj_to_origin(self, object_points, hand_joints):
        min_p = object_points.min(0)
        max_p = object_points.max(0)
        obj_center = (max_p + min_p) / 2.0
        # print("obj center", obj_center)
        return object_points - obj_center, hand_joints - obj_center, obj_center

    def scale_to_cm(self, data_dict):
        scale = 100.0
        data_dict['object_points'] = data_dict['object_points'] * scale
        data_dict['hand_joints'] = data_dict['hand_joints'] * scale

        # Hand verts
        data_dict['hand_verts'] = data_dict['hand_verts'] * scale
        return data_dict

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.
        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True
