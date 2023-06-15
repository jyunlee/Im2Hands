import os
import sys
import time
import torch
import shutil
import trimesh
import argparse
import pandas as pd
import numpy as np
import open3d as o3d

from tqdm import tqdm
from collections import defaultdict

from artihand import config, data
from artihand.checkpoints import CheckpointIO
from artihand.nasa.kpts_ref_training import preprocess_joints

from dependencies.halo.halo_adapter.transform_utils import xyz_to_xyz1

2
parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--config', type=str, help='Path to config file.', default='configs/kpts_ref/kpts_ref.yaml')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--latest', action='store_true', help='Use latest model instead of best.')
parser.add_argument('--subset', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset subset')
parser.add_argument('--out_dir', type=str, default='/data/hand_data/kpts_ref', help='Path to output directory to store intermediate results.')
parser.add_argument('--split_idx', type=int, default=0, help='Dataset split index. (-1: no split)')
parser.add_argument('--splits', type=int, default=1000, help='Dataset split index. (-1: no split)')


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/kpts_ref/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    if args.latest:
        generation_dir = generation_dir + '_latest'
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    batch_size = cfg['generation']['batch_size']
    input_type = cfg['data']['input_type']

    dataset = config.get_dataset(args.subset, cfg, splits=args.splits, split_idx=args.split_idx)

    # Model
    model = config.get_model(cfg, device=device, dataset=dataset)

    checkpoint_io = CheckpointIO(out_dir, model=model)
    if args.latest:
        checkpoint_io.load('best_model.pt')
    else:
        checkpoint_io.load(cfg['test']['model_file'])

    # Generator
    generator = config.get_generator(model, cfg, device=device)

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1, shuffle=False)

    # Statistics
    time_dicts = []

    # Generate
    model.eval()

    args.out_dir = os.path.join(args.out_dir, args.subset)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    eval_list = defaultdict(list)

    for i, data in enumerate(tqdm(test_loader)): 

        eval_dict = {}

        img, camera_params, mano_data, idx = data

        joints_gt = {'left': mano_data['left'].get('joints').to(device),
                     'right': mano_data['right'].get('joints').to(device)}

        joints = {'left': mano_data['left'].get('pred_joints').to(device),
                  'right': mano_data['right'].get('pred_joints').to(device)}

        root_rot_mat = {'left': mano_data['left'].get('root_rot_mat').to(device),
                        'right': mano_data['left'].get('root_rot_mat').to(device)}

        kwargs = {}

        # joint space conversion & normalization
        left_joints, right_joints, left_norm = preprocess_joints(joints['left'], joints['right'], camera_params, root_rot_mat, return_mid=True)
        left_joints_gt, right_joints_gt = preprocess_joints(joints_gt['left'], joints_gt['right'], camera_params, root_rot_mat)

        in_joints = {'left': left_joints, 'right': right_joints}

        with torch.no_grad():
            left_joints_pred, right_joints_pred = model(img, camera_params, in_joints, **kwargs) 

        left_joints = left_joints_pred/1000 + left_norm
        right_joints = right_joints_pred/1000 + left_norm

        left_joints = torch.bmm(left_joints, camera_params['R'].double().cuda())
        left_joints = left_joints + camera_params['left_root_xyz'].cuda().unsqueeze(1) * torch.Tensor([-1., 1., 1.]).cuda()
        right_joints = torch.bmm(right_joints, camera_params['R'].double().cuda())
        right_joints = right_joints + camera_params['right_root_xyz'].cuda().unsqueeze(1)

        left_joints_out_path = os.path.join(args.out_dir, '%07d_left.ply' % idx)
        right_joints_out_path = os.path.join(args.out_dir, '%07d_right.ply' % idx)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(left_joints.detach().cpu().numpy()[0])
        o3d.io.write_point_cloud(left_joints_out_path, pcd)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(right_joints.detach().cpu().numpy()[0])
        o3d.io.write_point_cloud(right_joints_out_path, pcd)


