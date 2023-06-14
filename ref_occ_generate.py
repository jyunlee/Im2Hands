import os
import sys
import time
import torch
import shutil
import trimesh
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from artihand import config, data
from artihand.checkpoints import CheckpointIO

from dependencies.halo.halo_adapter.transform_utils import xyz_to_xyz1


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--config', type=str, help='Path to config file.', default='configs/ref_occ/ref_occ.yaml')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--latest', action='store_true', help='Use latest model instead of best.')
parser.add_argument('--subset', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset subset')
parser.add_argument('--out_dir', type=str, default='/data/hand_data/ref_occ_vis', help='Path to output directory to store intermediate results.')
parser.add_argument('--split_idx', type=int, default=0, help='Dataset split index. (-1: no split)')
parser.add_argument('--splits', type=int, default=1000, help='Dataset split index. (-1: no split)')


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/ref_occ/default.yaml')
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

    # Determine what to generate
    generate_mesh = cfg['generation']['generate_mesh']

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1, shuffle=False)

    # Statistics
    time_dicts = []

    # Generate
    model.eval()

    mesh_dir = os.path.join(generation_dir, 'meshes')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )

    args.out_dir = os.path.join(args.out_dir, args.subset)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i, data in enumerate(tqdm(test_loader)): 

        img, camera_params, mano_data, idx = data

        # Create directories if necessary
        if generate_mesh and not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        
        # Generate outputs
        out_file_dict = {}

        if generate_mesh:
            out = generator.ref_occ_generate_mesh(data)
            
            # Get statistics
            left_mesh, right_mesh = out

            # Write output
            left_mesh_out_path = os.path.join(args.out_dir, '%07d_left.obj' % idx)
            right_mesh_out_path = os.path.join(args.out_dir, '%07d_right.obj' % idx)

            print(f"Generating {left_mesh_out_path}...")
            print(f"Generating {right_mesh_out_path}...")

            os.makedirs(os.path.dirname(left_mesh_out_path), exist_ok=True)

            left_mid_joint = torch.matmul(mano_data['left']['root_rot_mat'].squeeze().cpu(), xyz_to_xyz1(mano_data['left']['mid_joint'] * torch.Tensor([-1., 1., 1.])).unsqueeze(-1).float())[:, :3, 0]
            right_mid_joint = torch.matmul(mano_data['right']['root_rot_mat'].squeeze().cpu(), xyz_to_xyz1(mano_data['right']['mid_joint']).unsqueeze(-1).float())[:, :3, 0]

            # Save left hand mesh
            left_mesh.vertices = left_mesh.vertices * 0.4 
            #left_mesh.vertices = left_mesh.vertices - left_mid_joint.cpu().numpy()
            left_mesh.vertices = torch.matmul(mano_data['left']['root_rot_mat'].squeeze().cpu().T, xyz_to_xyz1(torch.Tensor(left_mesh.vertices)).unsqueeze(-1))[:, :3, 0]
            left_mesh.vertices = left_mesh.vertices + camera_params['left_root_xyz'].cpu().numpy()
            left_mesh.vertices = left_mesh.vertices * [-1, 1, 1] 

            trimesh.repair.fix_inversion(left_mesh)
            left_mesh.export(left_mesh_out_path)

            # Save right hand mesh
            right_mesh.vertices = right_mesh.vertices * 0.4 
            #right_mesh.vertices = right_mesh.vertices - right_mid_joint.cpu().numpy()
            right_mesh.vertices = torch.matmul(mano_data['right']['root_rot_mat'].squeeze().cpu().T, xyz_to_xyz1(torch.Tensor(right_mesh.vertices)).unsqueeze(-1))[:, :3, 0]
            right_mesh.vertices = right_mesh.vertices + camera_params['right_root_xyz'].cpu().numpy()
            right_mesh.export(right_mesh_out_path)

            import cv2
            img = cv2.imread(camera_params['img_path'][0])
            cv2.imwrite(os.path.join(args.out_dir, '%07d_img.png' % idx), img)


