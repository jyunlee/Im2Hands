import torch
import torch.optim as optim
from torch import autograd
import numpy as np
import os
from tqdm import trange
import trimesh
from trimesh.base import Trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time

from models.utils import visualize as vis
from models.data.input_helpers import rot_mat_by_angle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# mano loss
import sys
sys.path.insert(0, "/home/korrawe/halo_vae/scripts")
from manopth.manolayer import ManoLayer
from manopth import demo


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

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
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

        # MANO
        self.mano_layer = ManoLayer(
            mano_root='/home/korrawe/halo_vae/scripts/mano/models', center_idx=0, use_pca=True, ncomps=45, flat_hand_mean=False)
        self.mano_layer = self.mano_layer.to(device)

    def sample_keypoint(self, data, obj_idx, N=1, gen_mesh=False, mesh_dir=None, 
                        random_rotate=False, return_stats=True, use_refine_net=False):
        ''' Sample n hands for the given data.
        Args:
            data (dict): data dictionary
            N (int): number of hand to sample
            mesh_dir(str): output mesh location.
            return_stats (bool): whether stats should be returned
        '''
        sample_per_obj = 10  # 5 for obman  # 20 for grab
        self.model.eval()
        device = self.device
        stats_dict = {}
        mesh_out = []
        kps_out = []

        object_points = data.get('object_points').float().to(device)
        hand_joints_gt = data.get('hand_joints').float().to(device)

        rot_mat = data['rot_mat'][0].float().numpy()

        # Use BPS
        if self.model.use_bps:
            object_inputs = data.get('object_bps').float().to(device)
        else:
            object_inputs = object_points

        vis_idx = 0  # np.random.randint(64)
        object_points = object_points[vis_idx].unsqueeze(0)
        hand_joints_gt = hand_joints_gt[vis_idx].unsqueeze(0)

        obj_mesh_path = data['mesh_path'][0]
        obj_name = os.path.splitext(os.path.basename(obj_mesh_path))[0]

        # Ground truth object
        gt_object_points = Trimesh(vertices=object_points.detach().cpu().numpy()[0])
        if self.model.use_bps:
            gt_object_points.vertices *= 100.0
        # Rotate to canonical
        gt_object_points.vertices = gt_object_points.vertices + data['obj_center'][0].numpy()
        gt_obj_verts = np.expand_dims(gt_object_points.vertices, -1)
        gt_object_points.vertices = np.matmul(rot_mat.T, gt_obj_verts).squeeze(-1)
        # gt_object_points.export(os.path.join(mesh_dir, str(obj_name) + '_' + str(obj_idx % sample_per_obj) + '_gt_obj_points.obj'))

        # Copy input mesh if available
        # import pdb; pdb.set_trace()
        if len(obj_mesh_path) > 0:
            gt_object_mesh = trimesh.load(obj_mesh_path, process=False)
            gt_object_mesh.vertices = gt_object_mesh.vertices * data['scale'][0].item()  #  + data['obj_center'][0].numpy()
            # gt_object_mesh.export(os.path.join(mesh_dir, 'obj' + str(obj_name) + '_gt_obj_mesh.obj'))  # obj_idx
            gt_object_mesh.export(os.path.join(mesh_dir, str(obj_name) + '_gt_obj_mesh.obj'))  # obj_idx

        for n in range(N):
            # Rotate object input
            # if random_rotate:
            #     x_angle = np.random.rand() * np.pi * 2.0
            #     y_angle = np.random.rand() * np.pi * 2.0
            #     z_angle = np.random.rand() * np.pi * 2.0
            #     rot_mat = rot_mat_by_angle(x_angle, y_angle, z_angle)
            # else:
            #     rot_mat = np.eye(3)
            # rot_mat_t = torch.from_numpy(rot_mat).float().to(device)
            # object_inputs = torch.matmul(rot_mat_t, object_inputs.unsqueeze(-1)).squeeze(-1)

            output_joints, object_latent = self.model(object_inputs, sample=True, reture_obj_latent=True)
            # MANO
            use_mano_loss = False  # True
            if use_mano_loss:
                rot, pose, shape, trans = output_joints[:, :3], output_joints[:, 3:48], output_joints[:, 48:58], output_joints[:, 58:61]
                output_verts, output_joints = self.mano_layer(torch.cat((rot, pose), 1), shape, trans)
                output_joints = output_joints / 10.0
                output_verts = output_verts / 10.0

            object_points_vis = object_points.detach().cpu().numpy()[0]  # [vis_idx]
            output_joints_vis = output_joints.detach().cpu().numpy()[0]  # [vis_idx]
            gt_joints = hand_joints_gt.detach().cpu().numpy()[0]  # [vis_idx]

            if use_refine_net:
                # Do refinement
                # import pdb; pdb.set_trace()
                output_joints_refine = self.model.refine(output_joints, object_latent, object_points, step=3)  # 3

                refine_output_joints_vis = output_joints_refine.detach().cpu().numpy()[0]  # [vis_idx]
                fig = plt.figure()
                ax = fig.gca(projection=Axes3D.name)
                vis.plot_skeleton_single_view(output_joints_vis, joint_order='mano', object_points=object_points_vis, ax=ax, color='r', show=False)
                vis.plot_skeleton_single_view(refine_output_joints_vis, joint_order='mano', ax=ax, color='b', show=False)
                # fig.show()
                output_path = os.path.join(mesh_dir, '..', 'vis_compare', '%s_%03d.png' % (obj_name, obj_idx % sample_per_obj))
                plt.savefig(output_path)
                # fig.close()

            kps_out.append(output_joints_vis)

            # output_path = os.path.join(mesh_dir, '..', 'vis', '%03d_%03d.png' % (obj_idx, n))
            output_path = os.path.join(mesh_dir, '..', 'vis', '%s_%03d.png' % (obj_name, obj_idx % sample_per_obj))
            # col = 'b' if n == 0 else 'g'
            col = 'g'
            vis.visualise_skeleton(output_joints_vis, object_points_vis, joint_order='mano', color=col, out_file=output_path, show=False)
            # print('vis out', output_path)
            # vis.visualise_skeleton(output_joints_vis, object_points_vis, joint_order='mano', color=col, show=True)
            # vis.visualise_skeleton(gt_joints, object_points_vis, joint_order='mano', color=col, show=True)
            # vis.visualise_skeleton(hand_joints_gt_test.detach().cpu().numpy()[0], object_points_vis, joint_order='biomech', color=col, show=True)

            # Save keypoints numpy
            kps_path = os.path.join(mesh_dir, '..', 'kps', '%s_%03d.npy' % (obj_name, obj_idx % sample_per_obj))
            with open(kps_path, 'wb') as kps_file:
                np.save(kps_file, output_joints_vis)

            # output_joints_vis = np.load(kps_path)
            # output_joints = torch.from_numpy(output_joints_vis).float().to(device).unsqueeze(0)

            # MANO
            if use_mano_loss:
                sample_name = '%s_h%03d' % (str(obj_name), obj_idx % sample_per_obj)

                mano_faces = self.mano_layer.th_faces.detach().cpu()
                output_mesh = trimesh.Trimesh(vertices=output_verts.detach().cpu().numpy()[0], faces=mano_faces)
                output_mesh.vertices = output_mesh.vertices + data['obj_center'][0].numpy()
                # Rotate output back to original object orientation
                verts = np.expand_dims(output_mesh.vertices, -1)
                output_mesh.vertices = np.matmul(rot_mat.T, verts).squeeze(-1)
                # import pdb; pdb.set_trace()

                # Debug rotation
                # gt_object_points = Trimesh(vertices=object_inputs.detach().cpu().numpy()[0])
                # gt_object_points.export(os.path.join(mesh_dir, sample_name + '_gt_obj_points.obj'))

                meshout_path = os.path.join(mesh_dir, sample_name)
                output_mesh.export(meshout_path + '.obj')
                continue

            # HALO
            if self.model.halo_adapter is not None and gen_mesh:
                # sample_name = 'obj%03d_h%03d' % (obj_idx, n)
                sample_name = '%s_h%03d' % (str(obj_name), obj_idx % sample_per_obj)
                # For BPS
                if self.model.use_bps:
                    output_joints *= 100.0
                output_mesh, normalized_kps = self.model.halo_adapter(output_joints, joint_order='mano', return_kps=True, original_position=True)

                # # Move to object canonical
                # print(data['obj_center'][0].numpy())
                output_mesh.vertices = output_mesh.vertices + data['obj_center'][0].numpy()
                # Rotate output back to original object orientation
                verts = np.expand_dims(output_mesh.vertices, -1)
                output_mesh.vertices = np.matmul(rot_mat.T, verts).squeeze(-1)
                # import pdb; pdb.set_trace()

                # Debug rotation
                # gt_object_points = Trimesh(vertices=object_inputs.detach().cpu().numpy()[0])
                # gt_object_points.export(os.path.join(mesh_dir, sample_name + '_gt_obj_points.obj'))

                meshout_path = os.path.join(mesh_dir, sample_name)
                output_mesh.export(meshout_path + '.obj')

                # Optimize translation
                optimize_trans = False  # True
                if optimize_trans:
                    # import pdb; pdb.set_trace()
                    inside_points = data.get('inside_points').float().to(device)
                    translation = self.model.halo_adapter.optimize_trans(inside_points, output_joints, gt_object_mesh, joint_order='mano')
                    # import pdb; pdb.set_trace()
                    translation = translation.detach().cpu().numpy()
                    # Add translation
                    new_verts = verts.squeeze(-1) + translation
                    new_verts = np.expand_dims(new_verts, -1)
                    # Rotate back
                    # import pdb; pdb.set_trace()
                    final_joints = output_joints_vis + translation
                    final_verts = np.matmul(rot_mat.T, new_verts).squeeze(-1)
                    output_mesh.vertices = final_verts

                    meshout_path = os.path.join(mesh_dir, sample_name)
                    output_mesh.export(meshout_path + '_refine.obj')

                    # Save keypoints numpy
                    kps_path = os.path.join(mesh_dir, '..', 'kps', '%s_%03d_refine.npy' % (obj_name, obj_idx % sample_per_obj))
                    with open(kps_path, 'wb') as kps_file:
                        np.save(kps_file, final_joints)

                # For debugging ground truth
                # output_mesh, normalized_kps = self.model.halo_adapter(hand_joints_gt, joint_order='mano', return_kps=True, original_position=True)

                # mesh_out.append(output_mesh)
                # vis.visualise_skeleton(normalized_kps.detach().cpu().numpy()[0], object_points_vis, color=col, title=sample_name, show=True)

                # gt_kps_mesh = Trimesh(vertices=hand_joints_gt.detach().cpu().numpy()[0])
                # gt_kps_mesh.export(os.path.join(mesh_dir, sample_name + '_gt_kps.obj'))

        if N == 1:
            return kps_out[0]  # mesh_out[0]
        return kps_out  # mesh_out

    def generate_mesh(self, data, return_stats=True):
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

        # Preprocess if requires
        if self.preprocessor is not None:
            t0 = time.time()
            with torch.no_grad():
                inputs = self.preprocessor(inputs)
            stats_dict['time (preprocess)'] = time.time() - t0

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0
        # print(c.size())

        # z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
        mesh = self.generate_from_latent(c, bone_lengths=bone_lengths, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, c=None, bone_lengths=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
        Args:
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
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

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()
                # values = self.eval_points(
                #     pointsf, z, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        # mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
        mesh = self.extract_mesh(value_grid, c, bone_lengths=bone_lengths, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, c=None, bone_lengths=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points 
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # occ_hat = self.model.decode(pi, z, c, **kwargs).logits
                occ_hat = self.model.decode(pi, c, bone_lengths=bone_lengths,**kwargs)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def eval_point_colors(self, p, c=None, bone_lengths=None):
        ''' Re-evaluates the outputted points from marching cubes for vertex colors.
        Args:
            p (tensor): points 
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        pointsf = torch.FloatTensor(p).to(self.device)
        p_split = torch.split(pointsf, self.points_batch_size)
        point_labels = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # occ_hat = self.model.decode(pi, z, c, **kwargs).logits
                _, label = self.model.decode(pi, c, bone_lengths=bone_lengths, return_model_indices=True)

            point_labels.append(label.squeeze(0).detach().cpu())
            # print("label", label[:40])

        label = torch.cat(point_labels, dim=0)
        label = label.detach().cpu().numpy()
        return label

    def extract_mesh(self, occ_hat, c=None, bone_lengths=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.occ_hat
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
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

        # Get point colors
        if self.with_color_labels:
            vert_labels = self.eval_point_colors(vertices, c, bone_lengths=bone_lengths)
            vertex_colors = self.bone_colors[vert_labels]

            # Convert the mesh vertice back to canonical pose using the trans matrix of the label
            # self.convert_to_canonical = False # True
            # convert_to_canonical = True
            if self.convert_to_canonical:
                vertices = self.convert_mesh_to_canonical(vertices, c, vert_labels)
                vertices = vertices # * 2.5 * 2.5
        else:
            vertex_colors = None

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            # normals = self.estimate_normals(vertices, z, c)
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
            # self.refine_mesh(mesh, occ_hat, z, c)
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