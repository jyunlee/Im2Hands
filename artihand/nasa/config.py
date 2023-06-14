import os
import torch
import torch.distributions as dist
from torch import nn

from artihand import data
from artihand import config
from artihand.nasa import models, training, generation
from artihand.nasa import init_occ_training, ref_occ_training, kpts_ref_training 

def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.
    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    decoder_c_dim = cfg['model']['decoder_c_dim']

    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    decoder_part_output = (decoder == 'piece_rigid' or decoder == 'piece_deform' or decoder == 'piece_deform_pifu')

    use_bone_length = cfg['model']['use_bone_length']

    decoder = models.decoder_dict[decoder](
        decoder_c_dim,
        use_bone_length=use_bone_length,
        **decoder_kwargs
    )

    if cfg['model']['type'] == 'init_occ' or cfg['model']['type'] == 'ref_occ':
        left_decoder = cfg['model']['decoder']
        left_decoder = models.decoder_dict[left_decoder](
            decoder_c_dim,
            use_bone_length=use_bone_length,
            add_feature_dim=19,
            add_feature_layer_idx=2,
            **decoder_kwargs
        )

        right_decoder = cfg['model']['decoder']
        right_decoder = models.decoder_dict[right_decoder](
            decoder_c_dim,
            use_bone_length=use_bone_length,
            add_feature_dim=19,
            add_feature_layer_idx=2,
            **decoder_kwargs
        )

        model = models.ArticulatedHandNetInitOcc(
            left_decoder, right_decoder, 
            device=device
        )

        if cfg['model']['type'] == 'ref_occ': 
            init_occ_estimator = model


    if cfg['model']['type'] == 'ref_occ':
        decoder_key = second_decoder = cfg['model']['decoder']

        model = models.ArticulatedHandNetRefOcc(
            init_occ_estimator, device=device
        )

    elif cfg['model']['type'] == 'kpts_ref':
        model = models.ArticulatedHandNetKptsRef(
            device=device
        )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.
    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    skinning_loss_weight = cfg['model']['skinning_weight']
    use_sdf = cfg['model']['use_sdf']

    if cfg['model']['type'] == 'init_occ':
        trainer = init_occ_training.Trainer(
            model, optimizer, skinning_loss_weight=skinning_loss_weight,
            device=device, input_type=input_type,
            threshold=threshold,
            eval_sample=cfg['training']['eval_sample'],
        )

    elif cfg['model']['type'] == 'ref_occ':
        trainer = ref_occ_training.Trainer(
            model, optimizer, skinning_loss_weight=skinning_loss_weight,
            device=device, input_type=input_type,
            threshold=threshold,
            eval_sample=cfg['training']['eval_sample'],
        )

    elif cfg['model']['type'] == 'kpts_ref':
        trainer = kpts_ref_training.Trainer(
            model, optimizer, skinning_loss_weight=skinning_loss_weight,
            device=device, input_type=input_type,
            threshold=threshold,
            eval_sample=cfg['training']['eval_sample'],
        )


    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.
    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        with_color_labels=cfg['generation']['vert_labels'],
        convert_to_canonical=cfg['generation']['convert_to_canonical'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.
    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    fields = {}
    fields['points'] = data.PointsField(
        cfg['data']['points_file'], points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields


def get_data_helpers(mode, cfg):
    ''' Returns the data fields.
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    with_transforms = cfg['model']['use_camera']
    
    helpers = {}
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            helpers['points_iou'] = data.PointsHelper(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        # if voxels_file is not None:
        #     fields['voxels'] = data.VoxelsField(voxels_file)

    return helpers


def get_data_transforms(mode, cfg):
    ''' Returns the data transform dict of callable.
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    transform_dict = {}
    if cfg['model']['use_sdf']:
        transform_dict['points'] = data.SubsamplePointcloud(cfg['data']['points_subsample'])
        # transform_dict['off_points'] = data.SubsampleOffPoint(cfg['data']['points_subsample'])
        transform_dict['off_points'] = data.SampleOffPoint(cfg['data']['points_subsample'])
    else:
        transform_dict['points'] = data.SubsamplePoints(cfg['data']['points_subsample'])

    if (cfg['model']['decoder'] == 'piece_rigid' or 
        cfg['model']['decoder'] == 'piece_deform' or
        cfg['model']['decoder'] == 'piece_deform_pifu'):
        transform_dict['mesh_points'] = data.SubsampleMeshVerts(cfg['data']['mesh_verts_subsample'])
        # transform_dict['reshape_occ'] = data.ReshapeOcc(cfg['data']['mesh_verts_subsample'])


    return transform_dict
