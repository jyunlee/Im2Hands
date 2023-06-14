import numpy as np
import torch

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data['points']
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out['points'] = points[indices, :]
        data_out['normals'] = normals[indices, :]
        return data_out


class SubsampleOffPoint(object):
    ''' Point subsampling transformation class.

    It subsamples the points that are not on the surface.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        off_points = data['off_points']

        indices = np.random.randint(off_points.shape[0], size=self.N)
        data_out['off_points'] = off_points[indices, :]

        return data_out


class SampleOffPoint(object):
    ''' Transformation class for sampling off-surface point inside the bounding box.

    It sample off surface points.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        off_surface_coords = np.random.uniform(-0.5, 0.5, size=(self.N, 3))
        data_out['off_points'] = torch.from_numpy(off_surface_coords).float()

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data['points']
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                'points': points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                'points': points,
                'occ': occ,
                'volume': volume,
            })
        return data_out


class SubsampleMeshVerts(object):
    ''' Mesh vertices subsampling transformation class.

    It subsamples the mesh vertices data along with theirs labels.

    Args:
        N (int): number of vertices to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        mesh_verts = data['mesh_verts']
        mesh_vert_labels = data['mesh_vert_labels']

        idx = np.random.randint(mesh_verts.shape[0], size=self.N)
        data_out.update({
            'mesh_verts': mesh_verts[idx, :],
            'mesh_vert_labels':  mesh_vert_labels[idx],
        })

        return data_out
    

class ReshapeOcc(object):
    ''' Occupancy vector transformation class.

    It reshapes the occupancy vector from .

    Args:
        N (int): number of vertices to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        mesh_verts = data['mesh_verts']
        mesh_vert_labels = data['mesh_vert_labels']

        idx = np.random.randint(mesh_verts.shape[0], size=self.N)
        data_out.update({
            'mesh_verts': mesh_verts[idx, :],
            'mesh_vert_labels':  mesh_vert_labels[idx],
        })

        return data_out