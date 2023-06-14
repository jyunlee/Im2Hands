import torch
import torch.distributions as dist
from torch import nn
import os
# from im2mesh.encoder import encoder_dict
# from im2mesh.onet import models, training, generation
# from im2mesh import data
# from im2mesh import config

from models import data
from models import config
from models.naive import models, training, generation


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
    # dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']  # hand dim
    # c_dim = cfg['model']['c_dim']
    decoder_dim = cfg['model']['decoder_dim'] 
    use_bps = cfg['model']['use_bps']
    use_refine_net = cfg['model']['use_refine_net']

    # decoder_kwargs = cfg['model']['decoder_kwargs']
    # encoder_kwargs = cfg['model']['encoder_kwargs']
    # encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    # decoder_part_output = (decoder == 'piece_rigid' or decoder == 'piece_deform')

    # use_bone_length=cfg['model']['use_bone_length']

    # decoder = models.decoder_dict[decoder](
    #     dim=dim, z_dim=z_dim, c_dim=c_dim,
    #     **decoder_kwargs
    # )
    # decoder = models.decoder_dict[decoder](
    #     decoder_c_dim,
    #     **decoder_kwargs
    # )
    # encoder = models.encoder_dict[encoder](
    #     encoder_c_dim,
    #     **encoder_kwargs
    # )

    # if encoder is not None:
    #     encoder = encoder_dict[encoder](
    #         c_dim=c_dim,
    #         **encoder_kwargs
    #     )
    # else:
    #     encoder = None

    object_dim = cfg['model']['object_dim']
    object_hidden_dim = cfg['model']['object_hidden_dim']
    if use_bps:
        obj_encoder = None
    else:
        obj_encoder = models.encoder_dict['pointnet'](
            c_dim=object_dim,
            hidden_dim=object_hidden_dim
        )
    # hand_encoder = models.encoder_dict['simple'](
    #     D_in=21*3, H=128, D_out=128
    # )
    encoder_dim = cfg['model']['encoder_dim']
    hand_encoder_latent = models.encoder_latent_dict['simple_latent'](
        c_dim=object_dim,
        z_dim=z_dim,
        dim=encoder_dim
    )

    mano_params_out = False  # True
    if mano_params_out:
        D_out = 3 + 45 + 10 + 3
        # rot, pose, shape, trans
    else:
        D_out = 63
    decoder = models.decoder_dict['simple'](
        D_in=object_dim + z_dim, H=decoder_dim, D_out=D_out,
        mano_params_out=mano_params_out
    )

    refine_model = None
    if use_refine_net:
        refine_model = models.RefineNet(
            in_dim=21, c_dim=object_dim, dim=128
        )

    p0_z = get_prior_z(cfg, device)
    model = models.HaloVAE(
        obj_encoder=obj_encoder,
        encoder_latent=hand_encoder_latent,
        decoder=decoder,
        p0_z=p0_z,
        use_bps=use_bps,
        refine_net=refine_model,
        device=device
    )

    # if z_dim != 0:
    #     encoder_latent = models.encoder_latent_dict[encoder_latent](
    #         dim=dim, z_dim=z_dim, c_dim=c_dim,
    #         **encoder_latent_kwargs
    #     )
    # else:
    #     encoder_latent = None

    # if encoder == 'idx':
    #     encoder = nn.Embedding(len(dataset), c_dim)
    # elif encoder is not None:
    #     encoder = encoder_dict[encoder](
    #         c_dim=c_dim,
    #         **encoder_kwargs
    #     )
    # else:
    #     encoder = None

    # p0_z = get_prior_z(cfg, device)
    # model = models.OccupancyNetwork(
    #     decoder, encoder, encoder_latent, p0_z, device=device
    # )

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
    use_refine_net = cfg['model']['use_refine_net']
    use_inter_loss = cfg['model']['use_inter_loss']
    use_mano_loss = cfg['model']['use_mano_loss']

    # skinning_loss_weight = cfg['model']['skinning_weight']
    kl_weight = cfg['model']['kl_weight']

    trainer = training.Trainer(
        model, optimizer, kl_weight=kl_weight,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        use_refine_net=use_refine_net,
        use_inter_loss=use_inter_loss,
        use_mano_loss=use_mano_loss
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
    transform_dict['points'] = data.SubsamplePoints(cfg['data']['points_subsample'])
    if (cfg['model']['decoder'] == 'piece_rigid' or 
        cfg['model']['decoder'] == 'piece_deform'):
        transform_dict['mesh_points'] = data.SubsampleMeshVerts(cfg['data']['mesh_verts_subsample'])
        # transform_dict['reshape_occ'] = data.ReshapeOcc(cfg['data']['mesh_verts_subsample'])


    return transform_dict
