import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import sys
import time
import argparse
import torch
import torch.optim as optim
import numpy as np
import matplotlib; matplotlib.use('Agg')

from torch.utils.tensorboard import SummaryWriter
from artihand import config, data
from artihand.checkpoints import CheckpointIO


# Arguments
parser = argparse.ArgumentParser(
    description='Train a deep structured implicit function model for hand reconstruction.'
)
parser.add_argument('--config', type=str, help='Path to config file.', default='configs/ref_occ/ref_occ.yaml')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/ref_occ/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()
ts = t0

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg, splits=2000)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, 
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)
model = model.to('cuda')

# Intialize training
npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-08, amsgrad=False, weight_decay=1e-5)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

kwargs = {
    'model': model,
    'optimizer': optimizer,
}
checkpoint_io = CheckpointIO(
    out_dir, initialize_from=cfg['model']['initialize_from'],
    initialization_file_name=cfg['model']['initialization_file_name'],
    **kwargs)

checkpoint_io.load('init_occ.pt', strict=True)
checkpoint_io.load('model_best.pt', strict=True) # WARNING!
load_dict = {}

epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1) - 1
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
metric_val_best = -1

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000,
                                      gamma=0.2, last_epoch=epoch_it)
logger = SummaryWriter(os.path.join(out_dir, 'logs_pen'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

while True:
    epoch_it += 1

    for batch in train_loader:
        scheduler.step()
        it += 1

        loss_dict = trainer.train_step(batch)
        loss = loss_dict['total']
        for k, v in loss_dict.items():
            logger.add_scalar('train/loss/%s' % k, v, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))
            print('time per batch: %.2f, total time: %.2f' 
                  % (time.time() - ts, time.time() - t0))
            ts = time.time()

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if it % backup_every == 0:
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
