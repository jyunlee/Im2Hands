import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time
from skimage import measure

class Generator3D(object):
    '''  Generator class for Occupancy Networks.
    It provides functions to generate the final mesh as well refining options.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        with_color_labels (bool): whether to assign part-color to the output mesh vertices
        convert_to_canonical (bool): whether to reconstruct mesh in canonical pose (for debugging)
        simplify_nfaces (int): number of faces the mesh should be simplified to
        preprocessor (nn.Module): preprocessor for inputs
    '''

    def __init__(self, model, points_batch_size=1000000,
                 threshold=0.45, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 with_color_labels=False,
                 convert_to_canonical=False,
                 simplify_nfaces=None,
                 preprocessor=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.sample = sample
        self.with_color_labels = with_color_labels
        self.convert_to_canonical = convert_to_canonical
        self.simplify_nfaces = simplify_nfaces
        self.preprocessor = preprocessor

        self.bone_colors = np.array([
            (119, 41, 191, 255), (75, 170, 46, 255), (116, 61, 134, 255), (44, 121, 216, 255), (250, 191, 216, 255), (129, 64, 130, 255),
            (71, 242, 184, 255), (145, 60, 43, 255), (51, 68, 187, 255), (208, 250, 72, 255), (104, 155, 87, 255), (189, 8, 224, 255),
            (193, 172, 145, 255), (72, 93, 70, 255), (28, 203, 124, 255), (131, 207, 80, 255)
            ], dtype=np.uint8
        )


    def init_occ_generate_mesh(self, data, threshold=0.5):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        img, camera_params, mano_data, idx = data

        inputs = {'left': mano_data['left'].get('inputs').to(device),
                  'right': mano_data['right'].get('inputs').to(device)}
        root_rot_mat = {'left': mano_data['left'].get('root_rot_mat').to(device),
                        'right': mano_data['right'].get('root_rot_mat').to(device)} 
        bone_lengths = {'left': mano_data['left'].get('bone_lengths').to(device),
                        'right': mano_data['right'].get('bone_lengths').to(device)}

        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img.cuda())

        img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1] 
        img_feat = torch.cat((hms_f, dp_f), 1)
        img_feat = self.model.image_final_layer(img_feat) 

        if threshold is None:
            threshold = self.threshold

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            left_values, right_values = self.init_occ_eval_points(img_feat, camera_params, inputs, (pointsf, pointsf), root_rot_mat, bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            left_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)
            right_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            left_points = left_mesh_extractor.query()
            right_points = right_mesh_extractor.query()

            while left_points.shape[0] != 0 or right_points.shape[0] != 0:
                # Query points
                left_pointsf = torch.FloatTensor(left_points).to(self.device)
                right_pointsf = torch.FloatTensor(right_points).to(self.device)

                # Normalize to bounding box
                left_pointsf = left_pointsf / left_mesh_extractor.resolution
                left_pointsf = box_size * (left_pointsf - 0.5)

                right_pointsf = right_pointsf / right_mesh_extractor.resolution
                right_pointsf = box_size * (right_pointsf - 0.5)
 
                # Evaluate model and update
                left_values, right_values = self.init_occ_eval_points(img_feat, camera_params, inputs, (left_pointsf, right_pointsf), root_rot_mat, bone_lengths=bone_lengths, **kwargs)

                left_values = left_values.cpu().numpy()
                right_values = right_values.cpu().numpy()

                left_values = left_values.astype(np.float64)
                right_values = right_values.astype(np.float64)

                left_mesh_extractor.update(left_points, left_values)
                right_mesh_extractor.update(right_points, right_values)

                left_points = left_mesh_extractor.query()
                right_points = right_mesh_extractor.query()

            left_value_grid = left_mesh_extractor.to_dense()
            right_value_grid = right_mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        left_mesh = self.extract_mesh(left_value_grid, inputs['left'], bone_lengths['left'], stats_dict=stats_dict, threshold=threshold)
        right_mesh = self.extract_mesh(right_value_grid, inputs['left'], bone_lengths['right'], stats_dict=stats_dict, threshold=threshold)

        return left_mesh, right_mesh


    def ref_occ_generate_mesh(self, data, threshold=0.45):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        img, camera_params, mano_data, idx = data

        inputs = {'left': mano_data['left'].get('inputs').to(device),
                  'right': mano_data['right'].get('inputs').to(device)}
        anchor_points = {'left': mano_data['left'].get('anchor_points').to(device), 
                         'right': mano_data['right'].get('anchor_points').to(device)}
        root_rot_mat = {'left': mano_data['left'].get('root_rot_mat').to(device),
                        'right': mano_data['right'].get('root_rot_mat').to(device)} 
        bone_lengths = {'left': mano_data['left'].get('bone_lengths').to(device),
                        'right': mano_data['right'].get('bone_lengths').to(device)}

        img = img.cuda()

        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img)

        hms_global = self.model.hms_global_layer(hms_fmaps[0]).squeeze(-1).squeeze(-1)
        dp_global = self.model.dp_global_layer(dp_fmaps[0]).squeeze(-1).squeeze(-1)
        img_global = torch.cat([hms_global, dp_global], 1)

        img_f = nn.functional.interpolate(img_fmaps[-1], size=[256, 256], mode='bilinear')
        hms_f = nn.functional.interpolate(hms_fmaps[-1], size=[256, 256], mode='bilinear')
        dp_f = nn.functional.interpolate(dp_fmaps[-1], size=[256, 256], mode='bilinear')

        img_feat = torch.cat((hms_f, dp_f), 1)
        img_feat = self.model.image_final_layer(img_feat)

        img_feat = (img_feat, img_global)

        if threshold is None:
            threshold = self.threshold

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            left_values, right_values = self.ref_occ_eval_points(img, img_feat, camera_params, inputs, (pointsf, pointsf), anchor_points, root_rot_mat, bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            left_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)
            right_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            left_points = left_mesh_extractor.query()
            right_points = right_mesh_extractor.query()

            while left_points.shape[0] != 0 or right_points.shape[0] != 0:
                # Query points
                left_pointsf = torch.FloatTensor(left_points).to(self.device)
                right_pointsf = torch.FloatTensor(right_points).to(self.device)

                # Normalize to bounding box
                left_pointsf = left_pointsf / left_mesh_extractor.resolution
                left_pointsf = box_size * (left_pointsf - 0.5)

                right_pointsf = right_pointsf / right_mesh_extractor.resolution
                right_pointsf = box_size * (right_pointsf - 0.5)
 
                # Evaluate model and update
                left_values, right_values = self.ref_occ_eval_points(img, img_feat, camera_params, inputs, (left_pointsf, right_pointsf), anchor_points, root_rot_mat, bone_lengths=bone_lengths, **kwargs)

                left_values = left_values.cpu().numpy()
                right_values = right_values.cpu().numpy()

                left_values = left_values.astype(np.float64)
                right_values = right_values.astype(np.float64)

                left_mesh_extractor.update(left_points, left_values)
                right_mesh_extractor.update(right_points, right_values)

                left_points = left_mesh_extractor.query()
                right_points = right_mesh_extractor.query()

            left_value_grid = left_mesh_extractor.to_dense()
            right_value_grid = right_mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        left_mesh = self.extract_mesh(left_value_grid, inputs['left'], bone_lengths['left'], stats_dict=stats_dict, threshold=threshold)
        right_mesh = self.extract_mesh(right_value_grid, inputs['left'], bone_lengths['right'], stats_dict=stats_dict, threshold=threshold)

        return left_mesh, right_mesh


    def generate_mesh(self, data, return_stats=True, threshold=None, pointcloud=False, return_intermediate=False, e2e=False):

        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        bone_lengths = data.get('bone_lengths')
        if bone_lengths is not None:
            bone_lengths = bone_lengths.to(device)
        kwargs = {}

        # Preprocess if requires # currently - none
        if self.preprocessor is not None:
            print('check - preprocess')
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs - this is actually identity function (input - output not changed)
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)

        stats_dict['time (encode inputs)'] = time.time() - t0
        #print(c.size())
        

        # z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        mesh = self.generate_from_latent(c, bone_lengths=bone_lengths, stats_dict=stats_dict, threshold=threshold, pointcloud=pointcloud, return_intermediate = return_intermediate, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh


    def generate_from_latent(self, c=None, bone_lengths=None, stats_dict={}, threshold=None, pointcloud=False, return_intermediate=False, side=None, **kwargs):
        ''' Generates mesh from latent.
        Args:
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        if threshold is None:
            threshold = self.threshold

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            # values = self.eval_points(pointsf, z, c, **kwargs).cpu().numpy()
            values = self.eval_points(pointsf, c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            # center = torch.FloatTensor([-0.15, 0.0, 0.0]).to(self.device)
            # box_size = 0.8

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                # import pdb; pdb.set_trace()

                values = self.eval_points(
                    pointsf, c, bone_lengths=bone_lengths, side=side, **kwargs).cpu().numpy()

                # import pdb; pdb.set_trace()
                # values = self.eval_points(
                #     pointsf, z, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        # mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)

        if return_intermediate:
            return value_grid

        if not pointcloud:
            mesh = self.extract_mesh(value_grid, c, bone_lengths=bone_lengths, stats_dict=stats_dict, threshold=threshold)
        else:
            mesh = self.extract_pointcloud(value_grid, c, bone_lengths=bone_lengths, stats_dict=stats_dict, threshold=threshold)

        return mesh

 
    def init_occ_eval_points(self, img_feat, camera_params, c, p, root_rot_mat, bone_lengths, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points 
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        left_p, right_p = p

        left_p_split = torch.split(left_p, self.points_batch_size)
        right_p_split = torch.split(right_p, self.points_batch_size)

        left_occ_hats = []
        right_occ_hats = []
        
        assert len(left_p_split) == len(right_p_split)

        for idx in range(len(right_p_split)):

            left_pi = left_p_split[idx]
            right_pi = right_p_split[idx]

            left_pi = left_pi.unsqueeze(0).to(self.device)
            right_pi = right_pi.unsqueeze(0).to(self.device)

            p = {'left': left_pi, 'right': right_pi}

            with torch.no_grad():
                left_occ_hat, right_occ_hat = self.model.decode(img_feat, camera_params, c, p, root_rot_mat, bone_lengths, **kwargs)

            left_occ_hats.append(left_occ_hat.squeeze(0).detach().cpu())
            right_occ_hats.append(right_occ_hat.squeeze(0).detach().cpu())

        left_occ_hat = torch.cat(left_occ_hats, dim=0)
        right_occ_hat = torch.cat(right_occ_hats, dim=0)
        
        return left_occ_hat, right_occ_hat
 

    def ref_occ_eval_points(self, img, img_feat, camera_params, c, p, anchor_points, root_rot_mat, bone_lengths, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points 
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        left_p, right_p = p

        left_p_split = torch.split(left_p, self.points_batch_size)
        right_p_split = torch.split(right_p, self.points_batch_size)

        left_occ_hats = []
        right_occ_hats = []
        
        assert len(left_p_split) == len(right_p_split)

        for idx in range(len(right_p_split)):

            left_pi = left_p_split[idx]
            right_pi = right_p_split[idx]

            left_pi = left_pi.unsqueeze(0).to(self.device)
            right_pi = right_pi.unsqueeze(0).to(self.device)

            p = {'left': left_pi, 'right': right_pi}

            with torch.no_grad():
                left_occ_hat, right_occ_hat = self.model(img, camera_params, c, p, anchor_points, root_rot_mat, bone_lengths, img_feat=img_feat, test=True, **kwargs)

            left_occ_hats.append(left_occ_hat.squeeze(0).detach().cpu())
            right_occ_hats.append(right_occ_hat.squeeze(0).detach().cpu())

        left_occ_hat = torch.cat(left_occ_hats, dim=0)
        right_occ_hat = torch.cat(right_occ_hats, dim=0)
        
        return left_occ_hat, right_occ_hat
 

    def extract_mesh(self, occ_hat, c=None, bone_lengths=None, stats_dict=dict(), threshold=None):
        ''' Extracts the mesh from the predicted occupancy grid.occ_hat
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding

        if threshold is None:
            threshold = self.threshold

        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0

        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5

        # Undo padding
        vertices -= 1

        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        if vertices.shape[0] == 0:
            mesh = trimesh.Trimesh(vertices, triangles)
            return mesh 

        # Get point colors
        if self.with_color_labels:
            vert_labels = self.eval_point_colors(vertices, c, bone_lengths=bone_lengths)
            vertex_colors = self.bone_colors[vert_labels]

            if self.convert_to_canonical:
                vertices = self.convert_mesh_to_canonical(vertices, c, vert_labels)
                vertices = vertices 
        else:
            vertex_colors = None

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               vertex_colors=vertex_colors, ##### add vertex colors
                            #    face_colors=face_colors, ##### try face color
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh
    

    def convert_mesh_to_canonical(self, vertices, trans_mat, vert_labels):
        ''' Converts the mesh vertices back to canonical pose using the input transformation matrices
        and the labels.
        Args:
            vertices (numpy array?): vertices of the mesh
            c (tensor): latent conditioned code c. Must be a transformation matices without projection.
            vert_labels (tensor): labels indicating which sub-model each vertex belongs to.
        '''
        # print(trans_mat.shape)
        # print(vertices.shape)
        # print(type(vertices))
        # print(vert_labels.shape)

        pointsf = torch.FloatTensor(vertices).to(self.device)
        # print("pointssf before", pointsf.shape)
        # [V, 3] -> [V, 4, 1]
        pointsf = torch.cat([pointsf, pointsf.new_ones(pointsf.shape[0], 1)], dim=1)
        pointsf = pointsf.unsqueeze(2)
        # print("pointsf", pointsf.shape)

        vert_trans_mat = trans_mat[0, vert_labels]
        # print(vert_trans_mat.shape)
        new_vertices = torch.matmul(vert_trans_mat, pointsf)

        vertices = new_vertices[:,:3].squeeze(2).detach().cpu().numpy()
        # print("return", vertices.shape)

        return vertices # new_vertices


    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.
        Args:
            vertices (numpy array): vertices of the mesh
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        # z, c = z.unsqueeze(0), c.unsqueeze(0)
        c = c.unsqueeze(0)

        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            # occ_hat = self.model.decode(vi, z, c).logits
            occ_hat = self.model.decode(vi, c)
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals


    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.
        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                # self.model.decode(face_point.unsqueeze(0), z, c).logits
                self.model.decode(face_point.unsqueeze(0), c)
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
