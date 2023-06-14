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
from artihand.nasa.e2e_training import dict2dev 
from artihand.nasa.models.core_joint_att import process_joints 
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


    def joint_att_generate_mesh(self, imgs, joints, root_z, data, return_stats=True, threshold=None, pointcloud=False):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        # extract image feature
        img, mask, dense, camera_params = imgs 
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img.cuda())
        left_joints, right_joints = joints
        left_root_z, right_root_z = root_z

        batch_size = img.shape[0]

        img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1] 

        device = 'cuda' 
        left_joint_ids = torch.arange(21, dtype=torch.long, device=device)
        left_joint_ids = left_joint_ids.unsqueeze(0).repeat(batch_size, 1)
        left_joint_id_embeddings = self.model.left_joint_id_embeddings(left_joint_ids)

        left_joint_pt_embeddings = self.model.left_joint_pt_embeddings(left_joints.transpose(1,2).float())

        left_joint_ft = torch.cat((left_joint_id_embeddings, left_joint_pt_embeddings.transpose(1,2)), 2)

        right_joint_ids = torch.arange(21, dtype=torch.long, device=device)
        right_joint_ids = right_joint_ids.unsqueeze(0).repeat(batch_size, 1) 
        right_joint_id_embeddings = self.model.right_joint_id_embeddings(right_joint_ids)

        right_joint_pt_embeddings = self.model.right_joint_pt_embeddings(right_joints.transpose(1,2).float())

        right_joint_ft = torch.cat((right_joint_id_embeddings, right_joint_pt_embeddings.transpose(1,2)), 2)

        img_ft = torch.cat((hms_f, dp_f), 1)
        img_ft = self.model.img_projection_layer(img_ft)

        left_joint_ft = self.model.img_ex_left(img_ft, left_joint_ft)
        right_joint_ft = self.model.img_ex_right(img_ft, right_joint_ft) 

        left_joint_res = self.model.left_joint_res_reg(left_joint_ft.transpose(1,2)) 
        right_joint_res = self.model.left_joint_res_reg(right_joint_ft.transpose(1,2))

        left_joints = left_joints + left_joint_res.transpose(1,2) 
        right_joints = right_joints + right_joint_res.transpose(1,2)

        left_joints_pred = left_joints
        right_joints_pred = right_joints 

        left_root_z, right_root_z = root_z
        left_c, left_bone_lengths, right_c, right_bone_lengths = process_joints(left_joints, right_joints, left_root_z, right_root_z, camera_params)

        img_f = nn.functional.interpolate(img_f, size=[256, 256], mode='bilinear')
        hms_f = nn.functional.interpolate(hms_f, size=[256, 256], mode='bilinear') 
        dp_f = nn.functional.interpolate(dp_f, size=[256, 256], mode='bilinear')  

        img_feat = torch.cat((hms_f, dp_f), 1)
        img_feat = self.model.image_final_layer(img_feat) 

        # extract other info
        #left_c = data['left_inputs']
        #right_c = data['right_inputs']
        c = (left_c, right_c)

        #left_bone_lengths = data['left_bone_lengths']
        #right_bone_lengths = data['right_bone_lengths']
        bone_lengths = (left_bone_lengths, right_bone_lengths)

        left_root_rot_mat = data['left_root_rot_mat']
        right_root_rot_mat = data['right_root_rot_mat']
        root_rot_mat = (left_root_rot_mat, right_root_rot_mat)

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
            left_values, right_values = self.joint_att_eval_points(img_feat, camera_params, root_rot_mat, (pointsf, pointsf), c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            left_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)
            right_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            left_points = left_mesh_extractor.query()
            right_points = right_mesh_extractor.query()

            # center = torch.FloatTensor([-0.15, 0.0, 0.0]).to(self.device)
            # box_size = 0.8

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
                # import pdb; pdb.set_trace()

                left_values, right_values = self.pifu_eval_points(
                    img_feat, camera_params, root_rot_mat, (left_pointsf, right_pointsf), c, bone_lengths=bone_lengths, **kwargs)

                left_values = left_values.cpu().numpy()
                right_values = right_values.cpu().numpy()

                # import pdb; pdb.set_trace()
                # values = self.eval_points(
                #     pointsf, z, c, **kwargs).cpu().numpy()
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

        # mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)

        #if return_intermediate:
        #    return value_grid

        if not pointcloud:
            left_mesh = self.extract_mesh(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_mesh(right_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        else:
            left_mesh = self.extract_pointcloud(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_pointcloud(righ_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        if return_stats:
            return left_mesh, right_mesh, stats_dict
        else:
            return left_mesh, right_mesh


    def shape_att_generate_mesh(self, imgs, data, return_stats=True, threshold=None, pointcloud=False):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        # extract image feature
        img, mask, dense, camera_params = imgs 
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img.cuda())

        img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1] 

        #img_f = nn.functional.interpolate(img_f, size=[256, 256], mode='bilinear')
        #hms_f = nn.functional.interpolate(hms_f, size=[256, 256], mode='bilinear') 
        #dp_f = nn.functional.interpolate(dp_f, size=[256, 256], mode='bilinear')  

        img_feat = torch.cat((hms_f, dp_f), 1)
        img_feat = self.model.image_final_layer(img_feat) 

        # extract other info
        left_c = data['left_inputs']
        right_c = data['right_inputs']
        c = (left_c, right_c)

        left_bone_lengths = data['left_bone_lengths']
        right_bone_lengths = data['right_bone_lengths']
        bone_lengths = (left_bone_lengths, right_bone_lengths)

        left_root_rot_mat = data['left_root_rot_mat']
        right_root_rot_mat = data['right_root_rot_mat']
        root_rot_mat = (left_root_rot_mat, right_root_rot_mat)


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
            left_values, right_values = self.shape_att_eval_points(img_feat, camera_params, root_rot_mat, (pointsf, pointsf), c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            left_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)
            right_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            left_points = left_mesh_extractor.query()
            right_points = right_mesh_extractor.query()

            # center = torch.FloatTensor([-0.15, 0.0, 0.0]).to(self.device)
            # box_size = 0.8

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
                # import pdb; pdb.set_trace()

                left_values, right_values = self.shape_att_eval_points(
                    img_feat, camera_params, root_rot_mat, (left_pointsf, right_pointsf), c, bone_lengths=bone_lengths, **kwargs)

                left_values = left_values.cpu().numpy()
                right_values = right_values.cpu().numpy()

                # import pdb; pdb.set_trace()
                # values = self.eval_points(
                #     pointsf, z, c, **kwargs).cpu().numpy()
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

        # mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)

        #if return_intermediate:
        #    return value_grid

        if not pointcloud:
            left_mesh = self.extract_mesh(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_mesh(right_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        else:
            left_mesh = self.extract_pointcloud(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_pointcloud(righ_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        if return_stats:
            return left_mesh, right_mesh, stats_dict
        else:
            return left_mesh, right_mesh

    def refinement_generate_mesh(self, data, return_stats=True, threshold=None, pointcloud=False):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        # extract image feature
        '''
        img, mask, dense, camera_params = imgs 
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img.cuda())

        img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1] 

        #img_f = nn.functional.interpolate(img_f, size=[256, 256], mode='bilinear')
        #hms_f = nn.functional.interpolate(hms_f, size=[256, 256], mode='bilinear') 
        #dp_f = nn.functional.interpolate(dp_f, size=[256, 256], mode='bilinear')  

        img_feat = torch.cat((hms_f, dp_f), 1)
        img_feat = self.model.image_final_layer(img_feat) 

        # extract other info
        left_c = data['left_inputs']
        right_c = data['right_inputs']
        c = (left_c, right_c)

        left_bone_lengths = data['left_bone_lengths']
        right_bone_lengths = data['right_bone_lengths']
        bone_lengths = (left_bone_lengths, right_bone_lengths)

        left_root_rot_mat = data['left_root_rot_mat']
        right_root_rot_mat = data['right_root_rot_mat']
        root_rot_mat = (left_root_rot_mat, right_root_rot_mat)
        '''

        img, mask, dense, camera_params, mano_data = data
        imgs = (img, mask, dense, camera_params)

        left_c = left_inputs = mano_data['left'].get('inputs').to(device)
        right_c = right_inputs = mano_data['right'].get('inputs').to(device)
        c = (left_inputs, right_inputs)

        left_anchor_points = mano_data['left'].get('anchor_points').to(device)
        right_anchor_points = mano_data['right'].get('anchor_points').to(device)

        left_root_rot_mat = mano_data['left'].get('root_rot_mat').to(device)
        right_root_rot_mat = mano_data['right'].get('root_rot_mat').to(device)
        root_rot_mat = (left_root_rot_mat, right_root_rot_mat)

        left_bone_lengths = mano_data['left'].get('bone_lengths').to(device)
        right_bone_lengths = mano_data['right'].get('bone_lengths').to(device)
        bone_lengths = (left_bone_lengths, right_bone_lengths)

        img = img.cuda()

        with torch.no_grad(): 
            hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img)

            img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1]

            img_f = nn.functional.interpolate(img_f, size=[256, 256], mode='bilinear')
            hms_f = nn.functional.interpolate(hms_f, size=[256, 256], mode='bilinear')
            dp_f = nn.functional.interpolate(dp_f, size=[256, 256], mode='bilinear')

            img_feat = torch.cat((hms_f, dp_f), 1)
            img_feat = self.model.image_final_layer(img_feat)
        
            ref_hms, ref_mask, ref_dp, ref_img_fmaps, ref_hms_fmaps, ref_dp_fmaps = self.model.refinement_image_encoder(img)

            ref_hms_global = self.model.refinement_hms_global_layer(ref_hms_fmaps[0]).squeeze(-1).squeeze(-1)
            ref_dp_global = self.model.refinement_dp_global_layer(ref_dp_fmaps[0]).squeeze(-1).squeeze(-1)

            ref_img_global = torch.cat([ref_hms_global, ref_dp_global], 1)

            ref_img_f, ref_hms_f, ref_dp_f = ref_img_fmaps[-1], ref_hms_fmaps[-1], ref_dp_fmaps[-1]

            ref_img_f = nn.functional.interpolate(ref_img_f, size=[256, 256], mode='bilinear')
            ref_hms_f = nn.functional.interpolate(ref_hms_f, size=[256, 256], mode='bilinear')
            ref_dp_f = nn.functional.interpolate(ref_dp_f, size=[256, 256], mode='bilinear')

            ref_img_feat = torch.cat((ref_hms_f, ref_dp_f), 1)
            ref_img_feat = self.model.refinement_image_final_layer(ref_img_feat)

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
            left_values, right_values = self.refinement_eval_points(img_feat, ref_img_feat, ref_img_global, camera_params, (left_root_rot_mat, right_root_rot), (pointsf, pointsf), (left_anchor_points, right_anchor_points), c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            left_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)
            right_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            left_points = left_mesh_extractor.query()
            right_points = right_mesh_extractor.query()

            # center = torch.FloatTensor([-0.15, 0.0, 0.0]).to(self.device)
            # box_size = 0.8

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
                # import pdb; pdb.set_trace()

                left_values, right_values = self.refinement_eval_points(
                    img_feat, ref_img_feat, ref_img_global, camera_params, (left_root_rot_mat, right_root_rot_mat), (left_pointsf, right_pointsf), (left_anchor_points, right_anchor_points), c, bone_lengths=bone_lengths, **kwargs)

        
        
                left_values = left_values.cpu().numpy()
                right_values = right_values.cpu().numpy()

                # import pdb; pdb.set_trace()
                # values = self.eval_points(
                #     pointsf, z, c, **kwargs).cpu().numpy()
                left_values = left_values.astype(np.float64)
                right_values = right_values.astype(np.float64)

                if left_points.shape[0] == 0:
                    left_values = np.zeros(0)
                if right_points.shape[0] == 0:
                    right_values = np.zeros(0)

                print(left_points.shape, left_values.shape)
                print(right_points.shape, right_values.shape)

                left_mesh_extractor.update(left_points, left_values)
                right_mesh_extractor.update(right_points, right_values)

                left_points = left_mesh_extractor.query()
                right_points = right_mesh_extractor.query()

            left_value_grid = left_mesh_extractor.to_dense()
            right_value_grid = right_mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        # mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)

        #if return_intermediate:
        #    return value_grid

        if not pointcloud:
            left_mesh = self.extract_mesh(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_mesh(right_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        else:
            left_mesh = self.extract_pointcloud(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_pointcloud(righ_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        if return_stats:
            return left_mesh, right_mesh, stats_dict
        else:
            return left_mesh, right_mesh




    def pifu_generate_mesh(self, imgs, data, return_stats=True, threshold=None, pointcloud=False):
        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        # extract image feature
        img, mask, dense, camera_params = imgs 
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.model.image_encoder(img.cuda())

        img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1] 

        img_f = nn.functional.interpolate(img_f, size=[256, 256], mode='bilinear')
        hms_f = nn.functional.interpolate(hms_f, size=[256, 256], mode='bilinear') 
        dp_f = nn.functional.interpolate(dp_f, size=[256, 256], mode='bilinear')  

        img_feat = torch.cat((hms_f, dp_f), 1)
        img_feat = self.model.image_final_layer(img_feat) 

        # extract other info
        left_c = data['left_inputs']
        right_c = data['right_inputs']
        c = (left_c, right_c)

        left_bone_lengths = data['left_bone_lengths']
        right_bone_lengths = data['right_bone_lengths']
        bone_lengths = (left_bone_lengths, right_bone_lengths)

        left_root_rot_mat = data['left_root_rot_mat']
        right_root_rot_mat = data['right_root_rot_mat']
        root_rot_mat = (left_root_rot_mat, right_root_rot_mat)


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
            left_values, right_values = self.pifu_eval_points(img_feat, camera_params, root_rot_mat, (pointsf, pointsf), c, bone_lengths=bone_lengths, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            left_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)
            right_mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            left_points = left_mesh_extractor.query()
            right_points = right_mesh_extractor.query()

            # center = torch.FloatTensor([-0.15, 0.0, 0.0]).to(self.device)
            # box_size = 0.8

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
                # import pdb; pdb.set_trace()

                left_values, right_values = self.pifu_eval_points(
                    img_feat, camera_params, root_rot_mat, (left_pointsf, right_pointsf), c, bone_lengths=bone_lengths, **kwargs)

                left_values = left_values.cpu().numpy()
                right_values = right_values.cpu().numpy()

                # import pdb; pdb.set_trace()
                # values = self.eval_points(
                #     pointsf, z, c, **kwargs).cpu().numpy()
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

        # mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)

        #if return_intermediate:
        #    return value_grid

        if not pointcloud:
            left_mesh = self.extract_mesh(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_mesh(right_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        else:
            left_mesh = self.extract_pointcloud(left_value_grid, left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold)
            right_mesh = self.extract_pointcloud(righ_value_grid, right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold)

        if return_stats:
            return left_mesh, right_mesh, stats_dict
        else:
            return left_mesh, right_mesh


    def e2e_generate_mesh(self, data, return_stats=True, threshold=None, pointcloud=False, return_intermediate=False, e2e=False):

        ''' Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        left_c = data['left_inputs']
        right_c = data['right_inputs']


        left_mesh = self.generate_from_latent(left_c, bone_lengths=left_bone_lengths, stats_dict=stats_dict, threshold=threshold, pointcloud=pointcloud, return_intermediate = return_intermediate, side = 'left', **kwargs)

        right_mesh = self.generate_from_latent(right_c, bone_lengths=right_bone_lengths, stats_dict=stats_dict, threshold=threshold, pointcloud=pointcloud, return_intermediate = return_intermediate, side = 'right', **kwargs)

        if return_stats:
            return left_joints, right_joints, left_mesh, right_mesh, left_rot_then_swap_mat, right_rot_then_swap_mat, stats_dict
        else:
            return left_joints, right_joints, left_mesh, right_mesh, left_rot_then_swap_mat, right_rot_then_swap_mat


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



    def joint_att_eval_points(self, img_feat, camera_params, root_rot_mat, p, c=None, bone_lengths=None, side=None, **kwargs):
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
        
        #for pi in p_split:
        assert len(left_p_split) == len(right_p_split)

        for idx in range(len(right_p_split)):

            left_pi = left_p_split[idx]
            right_pi = right_p_split[idx]

            left_pi = left_pi.unsqueeze(0).to(self.device)
            right_pi = right_pi.unsqueeze(0).to(self.device)

            with torch.no_grad():
                left_occ_hat, right_occ_hat = self.model.decode(img_feat, camera_params,root_rot_mat, (left_pi, right_pi), c, bone_lengths=bone_lengths, test=False, **kwargs)

                # If use SDF, flip the sign of the prediction so that the MISE works
                # import pdb; pdb.set_trace()
                if self.model.use_sdf:
                    print('Not Implemented')
                    exit()
                    occ_hat = -1 * occ_hat

            left_occ_hats.append(left_occ_hat.squeeze(0).detach().cpu())
            right_occ_hats.append(right_occ_hat.squeeze(0).detach().cpu())

        left_occ_hat = torch.cat(left_occ_hats, dim=0)
        right_occ_hat = torch.cat(right_occ_hats, dim=0)
        
        return left_occ_hat, right_occ_hat
 

    def shape_att_eval_points(self, img_feat, camera_params, root_rot_mat, p, c=None, bone_lengths=None, side=None, **kwargs):
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
        
        #for pi in p_split:
        assert len(left_p_split) == len(right_p_split)

        for idx in range(len(right_p_split)):

            left_pi = left_p_split[idx]
            right_pi = right_p_split[idx]

            left_pi = left_pi.unsqueeze(0).to(self.device)
            right_pi = right_pi.unsqueeze(0).to(self.device)

            with torch.no_grad():
                left_occ_hat, right_occ_hat, _, _ = self.model.decode(img_feat, camera_params,root_rot_mat, (left_pi, right_pi), c, bone_lengths=bone_lengths, test=False, **kwargs)

                # If use SDF, flip the sign of the prediction so that the MISE works
                # import pdb; pdb.set_trace()
                if self.model.use_sdf:
                    print('Not Implemented')
                    exit()
                    occ_hat = -1 * occ_hat

            left_occ_hats.append(left_occ_hat.squeeze(0).detach().cpu())
            right_occ_hats.append(right_occ_hat.squeeze(0).detach().cpu())

        left_occ_hat = torch.cat(left_occ_hats, dim=0)
        right_occ_hat = torch.cat(right_occ_hats, dim=0)
        
        return left_occ_hat, right_occ_hat
 

    def refinement_eval_points(self, img_feat, ref_img_feat, ref_img_global, camera_params, root_rot_mat, p, anchor_points, c=None, bone_lengths=None, side=None, **kwargs):
        ''' Evaluates the occupancy values for the points.
        Args:
            p (tensor): points 
            # z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        left_p, right_p = p
        left_c, right_c = c 

        left_p_split = torch.split(left_p, self.points_batch_size)
        right_p_split = torch.split(right_p, self.points_batch_size)

        left_anchor_points, right_anchor_points = anchor_points
        left_root_rot_mat, right_root_rot_mat = root_rot_mat

        left_occ_hats = []
        right_occ_hats = []
        
        #for pi in p_split:
        assert len(left_p_split) == len(right_p_split)

        for idx in range(len(right_p_split)):

            left_pi = left_p_split[idx]
            right_pi = right_p_split[idx]

            left_pi = left_pi.unsqueeze(0).to(self.device)
            right_pi = right_pi.unsqueeze(0).to(self.device)

            with torch.no_grad():
                left_p_r, right_p_r, sub_left_img_feat, sub_right_img_feat = self.model.org_decode(img_feat, ref_img_feat, camera_params, root_rot_mat, (left_pi, right_pi), c, bone_lengths=bone_lengths, test=True, **kwargs)

                left_anchor_img_feat = self.model.anchor_to_img_feat(img_feat, left_anchor_points, left_root_rot_mat, camera_params, 'left')
                right_anchor_img_feat = self.model.anchor_to_img_feat(img_feat, right_anchor_points, right_root_rot_mat, camera_params, 'right')

                left_labels = torch.FloatTensor([1, 0]).unsqueeze(0).repeat_interleave(left_anchor_img_feat.shape[0], dim=0).cuda()
                left_labels = left_labels.unsqueeze(1).repeat_interleave(left_anchor_img_feat.shape[1], 1)

                left_anchor_feat = torch.cat([left_anchor_img_feat, left_labels], 2)

                right_labels = torch.FloatTensor([0, 1]).unsqueeze(0).repeat_interleave(right_anchor_img_feat.shape[0], dim=0).cuda()
                right_labels = right_labels.unsqueeze(1).repeat_interleave(right_anchor_img_feat.shape[1], 1)

                right_anchor_feat = torch.cat([right_anchor_img_feat, right_labels], 2)

                left_anchor_points = self.model.anchor_to_img_coord(left_anchor_points, left_root_rot_mat, camera_params, 'left')
                right_anchor_points = self.model.anchor_to_img_coord(right_anchor_points, right_root_rot_mat, camera_params, 'right')

                min_xyz = torch.min(torch.cat([left_anchor_points, right_anchor_points], 1), 1)[0] 
                max_xyz = torch.max(torch.cat([left_anchor_points, right_anchor_points], 1), 1)[0]

                center_xyz = (max_xyz.unsqueeze(1) + min_xyz.unsqueeze(1)) / 2

                left_anchor_points -= center_xyz
                right_anchor_points -= center_xyz 

                left_pt_feat = self.model.left_transformer_encoder(torch.cat((left_anchor_points, left_anchor_feat), 2))
                right_pt_feat = self.model.right_transformer_encoder(torch.cat((right_anchor_points, right_anchor_feat), 2))

                left_occ_hat, right_occ_hat = self.model.refinement_decode(img_feat, camera_params, root_rot_mat, (left_pi, right_pi), (left_p_r, right_p_r), c, (left_anchor_points, right_anchor_points), center_xyz, (sub_left_img_feat, sub_right_img_feat), (left_pt_feat, right_pt_feat), ref_img_global, bone_lengths=bone_lengths, test=True, **kwargs)

                left_occ_hat = left_occ_hat[0]
                right_occ_hat = right_occ_hat[0]

                if len(left_occ_hat.shape) == 0:
                    left_occ_hat = torch.Tensor((1, 0)).cuda()
                if len(right_occ_hat.shape) == 0:
                    right_occ_hat = torch.Tensor((1, 0)).cuda()

                # If use SDF, flip the sign of the prediction so that the MISE works
                # import pdb; pdb.set_trace()
                if self.model.use_sdf:
                    print('Not Implemented')
                    exit()
                    occ_hat = -1 * occ_hat

            left_occ_hats.append(left_occ_hat.squeeze(0).detach().cpu())
            right_occ_hats.append(right_occ_hat.squeeze(0).detach().cpu())

        left_occ_hat = torch.cat(left_occ_hats, dim=0)
        right_occ_hat = torch.cat(right_occ_hats, dim=0)

        return left_occ_hat, right_occ_hat
 

    def pifu_eval_points(self, img_feat, camera_params, root_rot_mat, p, c=None, bone_lengths=None, side=None, **kwargs):
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
        
        #for pi in p_split:
        assert len(left_p_split) == len(right_p_split)

        for idx in range(len(right_p_split)):

            left_pi = left_p_split[idx]
            right_pi = right_p_split[idx]

            left_pi = left_pi.unsqueeze(0).to(self.device)
            right_pi = right_pi.unsqueeze(0).to(self.device)

            with torch.no_grad():
                left_occ_hat, right_occ_hat = self.model.decode(img_feat, camera_params,root_rot_mat, (left_pi, right_pi), c, bone_lengths=bone_lengths, test=True, **kwargs)

                # If use SDF, flip the sign of the prediction so that the MISE works
                # import pdb; pdb.set_trace()
                if self.model.use_sdf:
                    print('Not Implemented')
                    exit()
                    occ_hat = -1 * occ_hat

            left_occ_hats.append(left_occ_hat.squeeze(0).detach().cpu())
            right_occ_hats.append(right_occ_hat.squeeze(0).detach().cpu())

        left_occ_hat = torch.cat(left_occ_hats, dim=0)
        right_occ_hat = torch.cat(right_occ_hats, dim=0)
        
        return left_occ_hat, right_occ_hat
 

    def eval_points(self, p, c=None, bone_lengths=None, side=None, **kwargs):
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
                if side == 'left':
                    occ_hat = self.model.left_decode(pi, c, bone_lengths=bone_lengths,**kwargs)
                elif side == 'right':
                    occ_hat = self.model.right_decode(pi, c, bone_lengths=bone_lengths,**kwargs)
                else:
                    occ_hat = self.model.decode(pi, c, bone_lengths=bone_lengths,**kwargs)

                # If use SDF, flip the sign of the prediction so that the MISE works
                # import pdb; pdb.set_trace()
                if self.model.use_sdf:
                    occ_hat = -1 * occ_hat

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
                # import pdb; pdb.set_trace()
                _, label = self.model.decode(pi, c, bone_lengths=bone_lengths, return_model_indices=True)
                
            point_labels.append(label.squeeze(0).detach().cpu())
            # print("label", label[:40])

        label = torch.cat(point_labels, dim=0)
        label = label.detach().cpu().numpy()
        return label
    
    def extract_pointcloud(self, occ_hat, c=None, bone_lengths=None, stats_dict=dict(), threshold=None):
        ''' Extracts the mesh from the predicted occupancy grid.occ_hat
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape

        box_size = 1 + self.padding
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        if threshold is None:
            threshold = self.threshold

        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        #vertices, triangles = libmcubes.marching_cubes(
        #    occ_hat_padded, threshold)
        
        sigma = 0.02
        vertices = np.argwhere((0.5-sigma < occ_hat) * (occ_hat < 0.5+sigma)).astype(np.float64)
        triangles = None

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
            return mesh  #  None

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
 



    def extract_mesh(self, occ_hat, c=None, bone_lengths=None, stats_dict=dict(), threshold=None):
        ''' Extracts the mesh from the predicted occupancy grid.occ_hat
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
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

        # Skimage marching cubes
        # # try:
        # if True:
        #     # value_grid = np.pad(value_grid, 1, "constant", constant_values=-1e6)
        #     value_grid = occ_hat_padded
        #     verts, faces, normals, unused_var = measure.marching_cubes_lewiner(
        #         value_grid, min(threshold, value_grid.max()))
        #     del normals
        #     verts -= 1
        #     verts /= np.array([
        #         value_grid.shape[0] - 3, value_grid.shape[1] - 3,
        #         value_grid.shape[2] - 3
        #     ],
        #                     dtype=np.float32)
        #     verts = 1.1 * (verts - 0.5)
        #     # verts = scale * (verts - 0.5)
        #     # verts = verts * gt_scale + gt_center
        #     faces = np.stack([faces[..., 1], faces[..., 0], faces[..., 2]], axis=-1)
        #     mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        #     # vertices = verts
        #     return mesh
        # # except:  # pylint: disable=bare-except
        # #     return None

        if vertices.shape[0] == 0:
            mesh = trimesh.Trimesh(vertices, triangles)
            return mesh  #  None

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
