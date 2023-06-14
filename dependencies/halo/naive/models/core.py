import torch
import torch.nn as nn
from torch import distributions as dist

from models.halo_adapter.adapter import HaloAdapter


class HaloVAE(nn.Module):
    ''' HALO VAE Network class.
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, obj_encoder=None, hand_encoder=None, decoder=None, encoder_latent=None,
                 p0_z=None, refine_net=None, use_bps=False, device=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.zeros(128, device=device),
                               torch.ones(128, device=device))

        self.halo_adapter = None
        self.decoder = decoder.to(device)

        self.use_bps = use_bps

        if refine_net is not None:
            self.refine_net = refine_net.to(device)
        else:
            self.refine_net = None

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if obj_encoder is not None:
            self.obj_encoder = obj_encoder.to(device)
        else:
            self.obj_encoder = None

        if hand_encoder is not None:
            self.hand_encoder = hand_encoder.to(device)
        else:
            self.hand_encoder = None

        self._device = device
        self.p0_z = p0_z

    def initialize_halo(self, halo_config_file, denoiser_pth=None):
        ''' Attach HALO model to the keypoint model.
        Args:
            halo_adapter (HaloAdapter): Adapter that is already initialized
        '''
        self.halo_adapter = HaloAdapter(halo_config_file, device=self._device, denoiser_pth=denoiser_pth)
        print('initialized halo')

    def forward(self, obj_points, hand_joints=None, sample=True, reture_obj_latent=False, **kwargs):
        ''' Performs a forward pass through the network.
        Args:
            obj_points (tensor): input object
            hand_joints (tensor): hand_joints, not used during inference
            sample (bool): whether to sample for z, ignored if hand_joints is not None
        '''
        batch_size = obj_points.size(0)
        object_latent = self.encode_objects(obj_points)
        # Ignore object
        # object_latent = object_latent * 0.0
        # print('c', object_latent)
        if hand_joints is not None:
            q_z = self.infer_z(hand_joints, object_latent, **kwargs)
            # print('q_z', q_z)
            z = q_z.mean
            # print('self.p0_z mean', self.p0_z.mean)
        else:
            z = self.get_z_from_prior((batch_size,), sample=sample)
        # print('z', z)
        # p_r = self.decode(p, z, c, **kwargs)
        # z = z * 0.  # ####
        p_r = self.decode(z, object_latent, **kwargs)
        if reture_obj_latent:
            return p_r, object_latent
        return p_r

    def compute_kl_divergence(self, obj_points, hand_joints, reture_obj_latent=False, **kwargs):
        ''' Computes the expectation lower bound.
        Args:
            hand_joints (tensor): sampled points
            # occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        '''
        object_latent = self.encode_objects(obj_points)
        q_z = self.infer_z(hand_joints, object_latent, **kwargs)
        z = q_z.rsample()
        pred = self.decode(z, object_latent, **kwargs)

        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        if reture_obj_latent:
            return kl, pred, object_latent
        return kl, pred

    # def compute_elbo(self, obj_points, hand_joints, **kwargs):
    #     ''' Computes the expectation lower bound.
    #     Args:
    #         hand_joints (tensor): sampled points
    #         # occ (tensor): occupancy values for p
    #         inputs (tensor): conditioning input
    #     '''
    #     object_latent = self.encode_objects(obj_points)
    #     q_z = self.infer_z(hand_joints, object_latent, **kwargs)
    #     z = q_z.rsample()
    #     p_r = self.decode(z, object_latent, **kwargs)

    #     # rec_error = -p_r.log_prob(occ).sum(dim=-1)
    #     rec_error = self.mse_loss(p_r, hand_joints)
    #     kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
    #     elbo = -rec_error - kl

    #     return elbo, rec_error, kl

    def encode_objects(self, inputs):
        ''' Encodes the input.
        Args:
            input (tensor): the input
        '''

        if self.obj_encoder is not None:
            c = self.obj_encoder(inputs)
        else:
            # Return inputs
            # c = torch.empty(inputs.size(0), 0)
            c = inputs

        return c

    def decode(self, z, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        # batch_size, points_size, p_dim = p.size()
        # p = p.reshape(batch_size * points_size, p_dim)

        p_r = self.decoder(z, c)
        return p_r

    def infer_z(self, joints, c, **kwargs):
        ''' Infers z.
        Args:
            joints (tensor): joints tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        if self.encoder_latent is not None:
            # print('not None')
            mean_z, logstd_z = self.encoder_latent(joints, c, **kwargs)
        else:
            batch_size = joints.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        # print("kkkk", mean_z, torch.exp(logstd_z))
        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        # import pdb;pdb.set_trace()
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.
        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
            # print(" -- sample -- ")
            # print(z)
        else:
            z = self.p0_z.mean.to(self._device)
            # print(" -- mean -- ")
            z = z.expand(*size, *z.size())
            # print(" -- sample --")
            # print(z)
        # import pdb; pdb.set_trace()
        # import matplotlib.pyplot as plt

        return z

    def to(self, device):
        ''' Puts the model to the device.
        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def refine(self, joints, obj_code, object_points, step=1):
        tips_idx = torch.Tensor([4, 8, 12, 16, 20]).long()
        cur_joints = joints
        for i in range(step):
            # import pdb; pdb.set_trace()
            # tips_dist = cal_tips_dist(joints, obj_points)
            pred_dist = torch.cdist(cur_joints, object_points)
            min_val, min_idx = torch.min(pred_dist, dim=2)
            # tips_dist = min_val[:, tips_idx]
            tips_dist = min_val
            out = self.refine_net(cur_joints, obj_code, tips_dist)
            cur_joints = out
        return cur_joints


# class ArticulatedHandNet(nn.Module):
#     ''' Occupancy Network class.
#     Args:
#         decoder (nn.Module): decoder network
#         encoder (nn.Module): encoder network
#         encoder_latent (nn.Module): latent encoder network
#         p0_z (dist): prior distribution for latent code z
#         device (device): torch device
#     '''

#     def __init__(self, decoder, encoder=None, encoder_latent=None, use_bone_length=False, per_part_output=False,
#                  p0_z=None, device=None):
#         super().__init__()
#         if p0_z is None:
#             p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

#         self.decoder = decoder.to(device)
#         self.use_bone_length = use_bone_length

#         # if encoder_latent is not None:
#         #     self.encoder_latent = encoder_latent.to(device)
#         # else:
#         #     self.encoder_latent = None

#         # If true, return predicted occupancies for each part-model
#         self.per_part_output = per_part_output

#         if encoder is not None:
#             self.encoder = encoder.to(device)
#         else:
#             self.encoder = None

#         self._device = device
#         # self.p0_z = p0_z

#     def forward(self, p, inputs, bone_lengths=None, sample=True, **kwargs):
#         ''' Performs a forward pass through the network.
#         Args:
#             p (tensor): sampled points
#             inputs (tensor): conditioning input
#             sample (bool): whether to sample for z
#         '''

#         c = self.encode_inputs(inputs)
#         # z = self.get_z_from_prior((batch_size,), sample=sample)
#         # p_r = self.decode(p, z, c, **kwargs)
#         p_r = self.decode(p, c, bone_lengths=bone_lengths, **kwargs)
#         return p_r

#     # def compute_elbo(self, p, occ, inputs, **kwargs):
#     #     ''' Computes the expectation lower bound.
#     #     Args:
#     #         p (tensor): sampled points
#     #         occ (tensor): occupancy values for p
#     #         inputs (tensor): conditioning input
#     #     '''
#     #     c = self.encode_inputs(inputs)
#     #     q_z = self.infer_z(p, occ, c, **kwargs)
#     #     z = q_z.rsample()
#     #     p_r = self.decode(p, z, c, **kwargs)

#     #     rec_error = -p_r.log_prob(occ).sum(dim=-1)
#     #     kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
#     #     elbo = -rec_error - kl

#     #     return elbo, rec_error, kl

#     def encode_inputs(self, inputs):
#         ''' Encodes the input.
#         Args:
#             input (tensor): the input
#         '''

#         if self.encoder is not None:
#             c = self.encoder(inputs)
#         else:
#             # Return inputs
#             # c = torch.empty(inputs.size(0), 0)
#             c = inputs

#         return c
    
#     def decode(self, p, c, bone_lengths=None, reduce_part=True, return_model_indices=False, **kwargs):
#         ''' Returns occupancy probabilities for the sampled points.
#         Args:
#             p (tensor): points
#             z (tensor): latent code z
#             c (tensor): latent conditioned code c
#             # joints (tensor): joint locations
#             reduce_part (bool): whether to reduce the last (sub-model) dimention for 
#                 part-base model (with max() or logSumExp()). Only considered if part-base model is used.
#                 True when training normal occupancy, and False when training skinning weight.
#             return_model_indices (bool): only for geration
#         '''
#         ############### Expand latent code to match the number of points here
#         # reshape to [batch x points, latent size]
#         # sdf_data = (samples.cuda()).reshape(
#                         #     num_samp_per_scene * scene_per_subbatch, 5 # 4
#                         # )
#         ##### repeat interleave
#         batch_size, points_size, p_dim = p.size()
#         p = p.reshape(batch_size * points_size, p_dim)
#         # print("c shape", c.size())
#         c = c.repeat_interleave(points_size, dim=0)
#         if bone_lengths is not None:
#             bone_lengths = bone_lengths.repeat_interleave(points_size, dim=0)
#         # print("c shape", c.size())
        
#         # True during testing
#         if return_model_indices:
#             # If part-labels are needed, get [batch x bones] probabilities from the model and find argmax externally
#             p_r = self.decoder(p, c, bone_lengths, reduce_part=False)
#             p_r = self.decoder.sigmoid(p_r)
#             if self.decoder.smooth_max:
#                 _, sub_model_indices = p_r.max(1, keepdim=True)
#                 # p_r = p_r.logsumexp(1, keepdim=True)
#                 weights = nn.functional.softmax(5.0 * p_r, dim=1)
#                 p_r = torch.sum(weights * p_r, dim=1)
#             else:
#                 p_r, sub_model_indices = p_r.max(1, keepdim=True)
#             # p_r = self.decoder.sigmoid(p_r)
#             sub_model_indices = sub_model_indices.reshape(batch_size, points_size)
#             return p_r, sub_model_indices

#         else:
#             p_r = self.decoder(p, c, bone_lengths, reduce_part=reduce_part)
#             p_r = self.decoder.sigmoid(p_r)
#             if reduce_part:
#                 if self.decoder.smooth_max:
#                     # p_r = p_r.logsumexp(1, keepdim=True)
#                     weights = nn.functional.softmax(5.0 * p_r, dim=1)
#                     p_r = torch.sum(weights * p_r, dim=1)
#                 else:
#                     p_r, _ = p_r.max(1, keepdim=True)
#                 p_r = p_r.reshape(batch_size, points_size)
#             # p_r = self.decoder.sigmoid(p_r) # 
        
#         # # From NASA original
#         # # Soft-Max Blending
#         # weights = tf.nn.softmax(hparams.soft_blend * x, axis=1)
#         # x = tf.reduce_sum(weights * x, axis=1)
#         # # #

#         # # print("p_r shape", p_r.size())
#         # # print("reduce part", reduce_part)
#         # if reduce_part:
#         #     if self.per_part_output:
#         #         # print("size in model ", p_r.size())
#         #         print("before max", p_r.shape)

#         #         # If smooth max is used, the returned p_r will be pre-sigmoid.
#         #         # The index is obtained using max, while the prob is obtained using logSumExp
#         #         if self.decoder.smooth_max:
#         #             _, sub_model_indices = p_r.max(1, keepdim=True)
#         #             p_r = p_r.logsumexp(1, keepdim=True)
#         #             p_r = nn.Sigmoid(p_r)
#         #         else:
#         #             p_r, sub_model_indices = p_r.max(1, keepdim=True)
#         #         print("after max", p_r.shape)
#         #         # If use logSumExp instead of max()
#         #         # p_r = p_r.logsumexp(1, keepdim=True)
#         #         # p_r = p_r.exp().sum(dim=1, keepdim=True) - (p_r.size(-1) - 1.)
#         #         # p_r = p_r.log()
#         #         # print(p_r[0])
#         #         # print(p_r.size())
#         #         sub_model_indices = sub_model_indices.reshape(batch_size, points_size)
            
#         #     # reshape back to [batch, points]
#         #     p_r = p_r.reshape(batch_size, points_size)
#         #     if self.per_part_output and return_model_indices:
#         #         return p_r, sub_model_indices
#         # else:
#         #     p_r = nn.Sigmoid(p_r)
#         return p_r

#     # def infer_z(self, p, occ, c, **kwargs):
#     #     ''' Infers z.
#     #     Args:
#     #         p (tensor): points tensor
#     #         occ (tensor): occupancy values for occ
#     #         c (tensor): latent conditioned code c
#     #     '''
#     #     if self.encoder_latent is not None:
#     #         mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
#     #     else:
#     #         batch_size = p.size(0)
#     #         mean_z = torch.empty(batch_size, 0).to(self._device)
#     #         logstd_z = torch.empty(batch_size, 0).to(self._device)

#     #     q_z = dist.Normal(mean_z, torch.exp(logstd_z))
#     #     return q_z

#     # def get_z_from_prior(self, size=torch.Size([]), sample=True):
#     #     ''' Returns z from prior distribution.
#     #     Args:
#     #         size (Size): size of z
#     #         sample (bool): whether to sample
#     #     '''
#     #     if sample:
#     #         z = self.p0_z.sample(size).to(self._device)
#     #     else:
#     #         z = self.p0_z.mean.to(self._device)
#     #         z = z.expand(*size, *z.size())

#     #     return z

#     def to(self, device):
#         ''' Puts the model to the device.
#         Args:
#             device (device): pytorch device
#         '''
#         model = super().to(device)
#         model._device = device
#         return model