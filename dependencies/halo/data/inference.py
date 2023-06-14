import os
import logging
from torch.utils import data
import numpy as np
import yaml
import pickle
import torch
import trimesh

from models.data.input_helpers import random_rotate, rot_mat_by_angle


logger = logging.getLogger(__name__)

class InferenceDataset(data.Dataset):
    ''' Dataset class for inference. Only object meshes are available
    '''

    def __init__(self, dataset_folder, input_helpers=None, split=None, sample_surface=True,
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
        self.random_rotate = random_rotate

        self.sample_surface = sample_surface
        self.pc_sample = 600

        # # Get all models
        split_file = os.path.join(dataset_folder, 'datalist.txt')
        with open(split_file, 'r') as f:
            self.object_names = f.read().strip().split('\n')

        # Use groundtruth rots and trans
        self.use_gt_rots = False
        object_names_full = []
        obj_rots = []
        obj_trans = []
        if self.use_gt_rots:
            for obj in self.object_names:
                rot_file = os.path.join(dataset_folder, '..', 'grab_test', os.path.splitext(obj)[0])
                with open(rot_file + '.pickle', 'rb') as p_file:
                    obj_meta = pickle.load(p_file)
                for idx in range(len(obj_meta['rotmat'])):
                    object_names_full.append(obj)
                    obj_rots.append(obj_meta['rotmat'][idx])
                    obj_trans.append(obj_meta['trans_obj'][idx])
        else:
            n_sample = 10  # for ho3d  # 5  for obman
            for obj in self.object_names:
                for i in range(n_sample):
                    object_names_full.append(obj)

        self.object_names = object_names_full
        self.obj_rots = obj_rots
        self.obj_trans = obj_trans

        self.use_inside_points = False
        if self.use_inside_points:
            # Load inside points
            sampled_point_path = '../data/grab/test_sample_vol.npz'
            sample_points = np.load(sampled_point_path)
            points_dict = {}
            for k in sample_points.files:
                points_dict[k] = sample_points[k]
            self.sample_points = points_dict

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.object_names)  # len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        filename = os.path.join(self.dataset_folder, self.object_names[idx])

        data = {}

        if self.sample_surface:
            input_mesh = trimesh.load(filename, process=False)
            surface_points = trimesh.sample.sample_surface(input_mesh, self.pc_sample)[0]
            surface_points = torch.from_numpy(surface_points).float()
        else:
            # surface_points = torch.from_numpy(load_points(filename)).float()
            pass

        if self.random_rotate:
            # data['object_points'], data['hand_joints'], data['rot_mat'] = random_rotate(data['object_points'], data['hand_joints'])
            x_angle = np.random.rand() * np.pi * 2.0
            y_angle = np.random.rand() * np.pi * 2.0
            z_angle = np.random.rand() * np.pi * 2.0
            data['rot_mat'] = rot_mat_by_angle(x_angle, y_angle, z_angle)
            # data['rot_mat'] = random_rotate(data['object_points'], data['hand_joints'])
        else:
            # data['rot_mat'] = self.obj_rots[idx]
            data['rot_mat'] = np.eye(3)

        rotated_surface_points = torch.matmul(torch.from_numpy(data['rot_mat']).float(), surface_points.unsqueeze(-1)).squeeze(-1)
        surface_points = rotated_surface_points

        data['mesh_path'] = filename
        data['hand_joints'] = np.zeros([21, 3])
        # data['rot_mat'] = np.eye(3)
        data['object_points'] = surface_points

        if self.use_bps:
            data['scale'] = 100.0
            data['obj_center'] = 0.0
            # import pdb; pdb.set_trace()
            bps_name = os.path.join(self.dataset_folder, 'bps_' + os.path.splitext(self.object_names[idx])[0] + '.npy')
            obj_bps = np.load(bps_name)
            data['object_bps'] = torch.from_numpy(obj_bps)
        else:
            data['scale'] = 100.0
            data['obj_center'] = np.array([0., 0., 0.])
            # data['object_points'], data['obj_center'] = self.obj_to_origin(surface_points)
            data['object_points'], data['obj_center'] = self.scale_to_cm(data['object_points'], data['obj_center'], data['scale'])

        # Get insdie points
        if self.use_inside_points:
            obj_name_no_ext = os.path.splitext(self.object_names[idx])[0]
            inside_points = self.sample_points[obj_name_no_ext]
            keep_idx = np.random.choice(len(inside_points), 2000, replace=False)  # 600
            inside_points = inside_points[keep_idx]
            # inside_points = np.inside_points * self.rot_mats[idx].T
            inside_points = np.matmul(data['rot_mat'], np.expand_dims(inside_points, -1)).squeeze(-1)
            data['inside_points'] = inside_points * 100.0 - data['obj_center']

        if self.return_idx:
            data['idx'] = idx

        return data

    def obj_to_origin(self, object_points):
        # obj_center = object_points.mean(axis=0)
        min_p, _ = object_points.min(0)
        max_p, _ = object_points.max(0)
        obj_center = (max_p + min_p) / 2.0
        return object_points - obj_center, obj_center

    def scale_to_cm(self, object_points, obj_center, scale):
        return object_points * scale, obj_center * scale

    def get_model_dict(self, idx):
        return self.object_names[idx]

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
