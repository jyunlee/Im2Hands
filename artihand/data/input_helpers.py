import os
import glob
import random
# from PIL import Image
import numpy as np
# import trimesh
# from im2mesh.data.core import Field
# from im2mesh.utils import binvox_rw

# InputHelpers
class InputHelper(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.
        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.
        Args:
            files: files
        '''
        raise NotImplementedError


# 3D Fields
class TransMatInputHelper(InputHelper):
    ''' TransMatInputHelper.
    It provides the helper to load transformation matrix and point data. 
    This is used for the points randomly sampled in the bounding volume of the 3D shape.
    Args:
        file_ext (str): file type extension (e.g .npz)
        points_folder (str): name of folder containing the point
        use_bone_length (bool): whether bone lengths should be provided
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided
    '''
    def __init__(self, file_ext, points_folder='points', use_bone_length=False, transform=None, 
                 with_transforms=False, unpackbits=False):
        self.file_ext = file_ext
        self.points_folder = points_folder
        self.use_bone_length = use_bone_length
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, model_name, idx=None):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            model_name (str): index of category
            idx (int): ID of data point
        '''

        meta_file_path = os.path.join(model_path, 'meta', model_name + self.file_ext)
        meta_dict = np.load(meta_file_path)

        file_path = os.path.join(model_path, self.points_folder, model_name + self.file_ext)
        #print(file_path)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        verts = points_dict['verts'].astype(np.float32)
        vert_labels = points_dict['vert_labels'].astype(np.int32)
        joints_trans = points_dict['joints_trans'].astype(np.float32)
        # joints_trans = np.linalg.inv(joints_trans)
        joints = points_dict['hand_joints'].astype(np.float32)
        # joints = points_dict['hand_joints'][:16].astype(np.float32)
        # import pdb; pdb.set_trace()

        root_xyz = meta_dict['root_xyz'].astype(np.float32)

        data = {
            'points': points,
            'occ': occupancies,
            # 'joints_trans': joints_trans,
            'inputs': joints_trans,
            'joints': joints,
            'root_xyz': root_xyz,
            'mesh_verts': verts,
            'mesh_vert_labels': vert_labels,
            'root_rot_mat': meta_dict['root_rot_mat']
        }
        # print(file_path)
        # for k in points_dict.files:
        #     print(k)
        # print(points_dict.keys())
        if self.use_bone_length:
            data['bone_lengths'] = points_dict['bone_lengths'].astype(np.float32) #### wrong 's'
            # data['bone_lengths'] = points_dict['bone_length'].astype(np.float32) #### wrong 's'

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data


# 3D Fields for SDF output
class TransMatInputHelperSdf(InputHelper):
    ''' TransMatInputHelper.
    It provides the helper to load transformation matrix and point data. 
    This is used for the points randomly sampled in the bounding volume of the 3D shape.
    Args:
        file_ext (str): file type extension (e.g .npz)
        points_folder (str): name of folder containing the point
        use_bone_length (bool): whether bone lengths should be provided
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided
    '''
    def __init__(self, file_ext, points_folder='points', pointcloud_floder='pointcloud', use_bone_length=False, 
                 transform=None, with_transforms=False, unpackbits=False):
        self.file_ext = file_ext
        self.points_folder = points_folder
        self.pointcloud_floder = pointcloud_floder
        self.use_bone_length = use_bone_length
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, model_name, idx=None):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            model_name (str): index of category
            idx (int): ID of data point
        '''
        file_path_meta = os.path.join(model_path, self.points_folder, model_name + self.file_ext)
        file_path_pointcloud = os.path.join(model_path, self.pointcloud_floder, model_name + self.file_ext)

        points_dict = np.load(file_path_pointcloud)
        meta_dict = np.load(file_path_meta)

        points = points_dict['points']
        normals = points_dict['normals']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
            normals = normals.astype(np.float32)
            normals += 1e-4 * np.random.randn(*normals.shape)
        else:
            points = points.astype(np.float32)
            normals = normals.astype(np.float32)
        
        # Sample off-surface point when called
        # off_points = meta_dict['points']
        # # Use only oniformly sampled points
        # half_size = off_points.shape[0] // 2
        # off_points = off_points[:half_size]
        # if off_points.dtype == np.float16:
        #     off_points = off_points.astype(np.float32)
        #     off_points += 1e-4 * np.random.randn(*off_points.shape)
        # else:
        #     off_points = off_points.astype(np.float32)

        verts = meta_dict['verts'].astype(np.float32)
        vert_labels = meta_dict['vert_labels'].astype(np.int32)
        joints_trans = meta_dict['joints_trans'].astype(np.float32)
        # joints_trans = np.linalg.inv(joints_trans)
        # joints = meta_dict['hand_joints'][:16].astype(np.float32)

        data = {
            'points': points,
            'normals': normals,
            'off_points': 0,  # off_points,  # off_points sampled from self.transform function
            'inputs': joints_trans,
            'mesh_verts': verts,
            'mesh_vert_labels': vert_labels
        }
        # print("off_points")
        # print(file_path)
        # for k in points_dict.files:
        #     print(k)
        # print(points_dict.keys())
        if self.use_bone_length:
            data['bone_lengths'] = meta_dict['bone_lengths'].astype(np.float32)

        if self.with_transforms:
            data['loc'] = meta_dict['loc'].astype(np.float32)
            data['scale'] = meta_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        print('check!!!')
        print(data.keys())

        return data


class PointsHelper(InputHelper):
    ''' Point Field.
    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.
    Args:
        file_ext (str): file type extension (e.g .npz)
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided
    '''
    def __init__(self, file_ext, points_folder='points', transform=None,
                 with_transforms=False, unpackbits=False):
        self.file_ext = file_ext
        self.points_folder = points_folder
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, model_name, idx=None):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            model_name (str): model name
            idx (int): ID of data point
        '''
        # file_path = os.path.join(model_path, self.file_name)
        file_path = os.path.join(model_path, self.points_folder, model_name + self.file_ext)
        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            'points': points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data


class PointCloudHelper(InputHelper):
    ''' Point cloud field.
    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.
    Args:
        file_name (str): file name
        file_ext (str): file type extension (e.g .npz)
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_ext, folder='pointcloud', transform=None, with_transforms=False):
        self.file_ext = file_ext
        self.folder = folder
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, model_name, idx=None):
        ''' Loads the data point.
        Args:
            model_path (str): path to model
            model_name (str): model name
            idx (int): ID of data point
        '''
        # file_path = os.path.join(model_path, self.file_name)
        file_path = os.path.join(model_path, self.folder, model_name + self.file_ext)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            'points': points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_ext in files)
        return complete
