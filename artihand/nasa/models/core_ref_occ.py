import sys
import torch
import torch.nn as nn

from torch import distributions as dist
from torch.nn.functional import grid_sample
from im2mesh.common import make_3d_grid

from dependencies.halo.halo_adapter.transform_utils import xyz_to_xyz1 

from dependencies.intaghand.models.encoder import ResNetSimple
from dependencies.intaghand.models.model_attn.img_attn import * 
from dependencies.intaghand.models.model_attn.self_attn import *

from dependencies.airnets.AIRnet import PointTransformerEncoderV2, PointTransformerDecoderOcc


def extract_local_img_feat(img_feat, camera_params, p, root_rot_mat, side='right', anchor=False, test=False): 

    if anchor:
        p = p / 1000
        p = p.float()

        if side == 'left':
            img_p = p + (camera_params[f'{side}_root_xyz'].cuda().unsqueeze(1)) * torch.Tensor([-1., 1., 1.]).cuda()
        else:
            img_p = p + (camera_params[f'{side}_root_xyz'].cuda().unsqueeze(1)) 

    else:
        p = p * 0.4

        img_p = torch.bmm(xyz_to_xyz1(p), root_rot_mat)[:, :, :3]
        img_p = img_p + (camera_params[f'{side}_root_xyz'].cuda().unsqueeze(1)) 

        if side == 'left':
            img_p = img_p * torch.Tensor([-1., 1., 1.]).cuda()

    img_coor_p = img_p = torch.bmm(img_p, camera_params['R'].transpose(1,2).cuda()) + camera_params['T'].cuda().unsqueeze(1) 
    img_p = torch.bmm(img_p * 1000, camera_params['camera'].transpose(1,2).cuda().float())

    proj_img_p = torch.zeros((img_p.shape[0], img_p.shape[1], 2)).cuda()

    for i in range(proj_img_p.shape[0]):
        proj_img_p[i] = img_p[i, :, :2] / img_p[i, :, 2:] 

    proj_img_p = (proj_img_p - 128) / 128
    sub_img_feat = grid_sample(img_feat, proj_img_p.unsqueeze(2)[:,:,:,:2], align_corners=True)[:, :, :, 0]
    sub_img_feat = sub_img_feat.permute(0,2,1)

    ''' # for debugging
    import cv2
    img = cv2.imread(camera_params['img_path'][0])

    vis_p = vis_p[0]
    for idx in range(vis_p.shape[0]):
        pt = vis_p[idx]
        if pt.min() < 0 or pt.max() > 255: continue
        img[int(pt[1]), int(pt[0])] = [255, 0, 255]

    if anchor:
        cv2.imwrite('debug/anchor_check_%s.jpg' % side, img)
    else:
        cv2.imwrite('debug/check_%s.jpg' % side, img)
    '''

    return sub_img_feat, img_coor_p * torch.Tensor([1., -1., -1.]).cuda()


class ArticulatedHandNetRefOcc(nn.Module):
    ''' Occupancy Network class.
    Args:
        device (device): torch device
    '''

    def __init__(self, init_occ_estimator, device=None):

        super().__init__()

        self.init_occ = init_occ_estimator

        self.image_encoder = ResNetSimple(model_type='resnet50',
                                          pretrained=True,
                                          fmapDim=[128, 128, 128, 128],
                                          handNum=2,
                                          heatmapDim=21)

        self.image_final_layer = nn.Conv2d(256, 32, 1) 
        self.hms_global_layer = nn.Conv2d(128, 128, 8)
        self.dp_global_layer = nn.Conv2d(128, 128, 8)

        self.trans_enc = PointTransformerEncoderV2(npoints_per_layer=[512, 256, 128], nneighbor=16, nneighbor_reduced=16, nfinal_transformers=3, d_transformer=256, d_reduced=120, full_SA=True, has_features=True)
        self.trans_dec = PointTransformerDecoderOcc(dim_inp=256, dim=200, nneigh=9, hidden_dim=32, return_feature=True)

        self.context_enc = nn.Sequential(nn.Linear(256*3, 256))

        self._device = device


    def forward(self, img, camera_params, inputs, p, anchor_points, root_rot_mat, bone_lengths, sample=True, pen=False, img_feat=None, test=False, **kwargs):
        ''' Performs a forward pass through the network.'''
        img = img.cuda()

        if img_feat is None:
            hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.image_encoder(img)

            hms_global = self.hms_global_layer(hms_fmaps[0]).squeeze(-1).squeeze(-1)
            dp_global = self.dp_global_layer(dp_fmaps[0]).squeeze(-1).squeeze(-1)
            img_global = torch.cat([hms_global, dp_global], 1)

            img_f = nn.functional.interpolate(img_fmaps[-1], size=[256, 256], mode='bilinear')
            hms_f = nn.functional.interpolate(hms_fmaps[-1], size=[256, 256], mode='bilinear')
            dp_f = nn.functional.interpolate(dp_fmaps[-1], size=[256, 256], mode='bilinear')

            img_feat = torch.cat((hms_f, dp_f), 1)
            img_feat = self.image_final_layer(img_feat)
        else:
            img_feat, img_global = img_feat

        left_query_img_feat, left_query_pts = extract_local_img_feat(img_feat, camera_params, p['left'], root_rot_mat['left'], side='left', test=test)
        right_query_img_feat, right_query_pts = extract_local_img_feat(img_feat, camera_params, p['right'], root_rot_mat['right'], side='right', test=test)

        query_pts = {'left': left_query_pts, 'right': right_query_pts}
        query_img_feat = {'left': left_query_img_feat, 'right': right_query_img_feat}
    
        # swap query points for penetration check later
        if pen:
            pen_query_pts = {'left': right_query_pts, 'right': left_query_pts}
            pen_query_img_feat = {'left': right_query_img_feat, 'right': left_query_img_feat}

        with torch.no_grad():
            left_p_r, right_p_r = self.init_occ(img, camera_params, inputs, p, root_rot_mat, bone_lengths, sample=sample)

            if pen:
                pen_left_p_r, pen_right_p_r = self.init_occ(img, camera_params, inputs, p, root_rot_mat, bone_lengths, sample=sample, pen=True)

        init_p_r = {'left': left_p_r, 'right': right_p_r}

        if pen:
            pen_init_p_r = {'left': pen_left_p_r, 'right': pen_right_p_r}

        left_anchor_img_feat, left_anchor_pts = extract_local_img_feat(img_feat, camera_params, anchor_points['left'], root_rot_mat['left'], side='left', anchor=True) 
        right_anchor_img_feat, right_anchor_pts = extract_local_img_feat(img_feat, camera_params, anchor_points['right'], root_rot_mat['right'], side='right', anchor=True) 

        left_labels = torch.FloatTensor([1, 0]).unsqueeze(0).repeat_interleave(left_anchor_img_feat.shape[0], dim=0).cuda() 
        left_labels = left_labels.unsqueeze(1).repeat_interleave(left_anchor_img_feat.shape[1], 1) 
        
        left_anchor_feat = torch.cat([left_anchor_img_feat, left_labels], 2)
        
        right_labels = torch.FloatTensor([0, 1]).unsqueeze(0).repeat_interleave(right_anchor_img_feat.shape[0], dim=0).cuda() 
        right_labels = right_labels.unsqueeze(1).repeat_interleave(right_anchor_img_feat.shape[1], 1) 
        right_anchor_feat = torch.cat([right_anchor_img_feat, right_labels], 2)

        # normalize
        min_xyz = torch.min(torch.cat([left_anchor_pts, right_anchor_pts], 1), 1)[0]
        max_xyz = torch.max(torch.cat([left_anchor_pts, right_anchor_pts], 1), 1)[0]

        center_xyz = (max_xyz.unsqueeze(1) + min_xyz.unsqueeze(1)) / 2 

        left_anchor_pts -= center_xyz
        right_anchor_pts -= center_xyz

        query_pts['left'] -= center_xyz 
        query_pts['right'] -= center_xyz 

        left_pt_feat = self.trans_enc(torch.cat((left_anchor_pts, left_anchor_feat), 2))
        right_pt_feat = self.trans_enc(torch.cat((right_anchor_pts, right_anchor_feat), 2))

        anchor_feat = {'left': left_pt_feat, 'right': right_pt_feat}

        ref_left_p_r, ref_right_p_r = self.decode(img_feat, camera_params, inputs, query_pts, query_img_feat, init_p_r, anchor_feat, img_global, root_rot_mat, bone_lengths, **kwargs)

        if pen:
            pen_ref_left_p_r, pen_ref_right_p_r = self.decode(img_feat, camera_params, inputs, pen_query_pts, pen_query_img_feat, pen_init_p_r, anchor_feat, img_global, root_rot_mat, bone_lengths, **kwargs)
            return ref_left_p_r, ref_right_p_r, pen_ref_left_p_r, pen_ref_right_p_r

        return ref_left_p_r, ref_right_p_r

   
    def decode(self, img_feat, camera_params, c, p, p_img_feat, init_p_r, anchor_feat, img_global_feat, root_rot_mat, bone_lengths, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.'''

        left_z = anchor_feat['left']['z']
        right_z = anchor_feat['right']['z']

        anchor_feat['left']['z'] = self.context_enc(torch.cat((left_z, right_z, img_global_feat), 1))
        anchor_feat['right']['z'] = self.context_enc(torch.cat((right_z, left_z, img_global_feat), 1))

        left_p_feat = torch.cat((p['left'], p_img_feat['left'], init_p_r['left'].unsqueeze(-1).repeat_interleave(32, dim=-1)), 2)
        right_p_feat = torch.cat((p['right'], p_img_feat['right'], init_p_r['right'].unsqueeze(-1).repeat_interleave(32, dim=-1)), 2)

        if left_p_feat.shape[1] > 0:
            left_res_occ = self.trans_dec(left_p_feat, anchor_feat['left']).squeeze(-1)
        else:
            left_res_occ = torch.Tensor((0)).cuda()

        if right_p_feat.shape[1] > 0:
            right_res_occ = self.trans_dec(right_p_feat, anchor_feat['right']).squeeze(-1)
        else:
            right_res_occ = torch.Tensor((0)).cuda()

        final_left_occ = torch.nn.functional.sigmoid(left_res_occ)
        final_right_occ = torch.nn.functional.sigmoid(right_res_occ)

        return final_left_occ, final_right_occ


    def to(self, device):
        ''' Puts the model to the device.
        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

