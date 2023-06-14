import os
import sys
from glob import glob
import logging
import torch
import torchvision.transforms
from torch.utils import data
import numpy as np
import yaml
import cv2 as cv 
import pickle
import json
from manopth.manolayer import ManoLayer


sys.path.append('/workspace/IntagHand')

from dataset.dataset_utils import IMG_SIZE, HAND_BBOX_RATIO, HEATMAP_SIGMA, HEATMAP_SIZE, cut_img

logger = logging.getLogger(__name__)


class PIFuSampleHandDataset(data.Dataset):

    def __init__(self, data_path, mano_dataset_folder, input_helpers, split=None,
                 no_except=True, transforms=None, return_idx=False, subset=1, split_idx=0, lower_bound=0, use_vis_idx=False):

        assert split in ['train', 'test', 'val']

        #if split == 'train':
        #    split = 'test'
        # !!! WARNING: TEMPORAL - SHOULD BE REMOVED !!!
        self.split = split
        self.subset = subset
        self.lower_bound = lower_bound
        self.use_vis_idx = use_vis_idx

        if self.use_vis_idx:
            self.vis_idx = np.load('/data/interhand2.6m/test/transfer_vis/vis_list.npy')

        #mano_path = get_mano_path()
        #self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
        #                   'left': ManoLayer(mano_path['left'], center_idx=None)}
        #fix_shape(self.mano_layer)

        self.split_idx=split_idx
        self.mano_dataset_folder = mano_dataset_folder
        self.input_helpers = input_helpers
        self.no_except = no_except
        self.transforms = transforms
        self.return_idx = return_idx

        self.data_path = data_path
        self.size = len(glob(os.path.join(data_path, split, 'anno', '*.pkl')))

        self.normalize_img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with open(os.path.join(mano_dataset_folder, split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
            self.annot_file = json.load(f)

        self.right_mano_layer = ManoLayer(
                mano_root='/workspace/mano_v1_2/models', use_pca=False, ncomps=45, flat_hand_mean=False)
        self.left_mano_layer = ManoLayer(
                mano_root='/workspace/mano_v1_2/models', use_pca=False, ncomps=45, flat_hand_mean=False, side='left')


    def __len__(self):
        if not self.use_vis_idx:
            return self.size // self.subset - self.lower_bound
        else:
            return self.vis_idx.shape[0]
        

    def __getitem__(self, idx):

        if not self.use_vis_idx:
            idx *= self.subset
            idx += self.split_idx
            idx += self.lower_bound 
        else:
            idx = self.vis_idx[idx]
        #print(os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        img = cv.imread(os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        hms = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}.jpg'.format(idx)))
        mask = cv.imread(os.path.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        dense = cv.imread(os.path.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

        img = cv.resize(img, (IMG_SIZE, IMG_SIZE)) 
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        mask = cv.resize(mask, (IMG_SIZE, IMG_SIZE)) 
        maskTensor = torch.tensor(cv.cvtColor(mask, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        maskTensor = maskTensor.permute(2, 0, 1)
        maskTensor = self.normalize_img(maskTensor)

        dense = cv.resize(dense, (IMG_SIZE, IMG_SIZE)) 
        denseTensor = torch.tensor(cv.cvtColor(dense, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        denseTensor = denseTensor.permute(2, 0, 1)
        denseTensor = self.normalize_img(denseTensor)

        '''
        hms = []
        for hand_type in ['left', 'right']:
            for hIdx in range(7):
                hm = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(idx, hIdx, hand_type)))
                hm = cv.resize(hm, (img.shape[1], img.shape[0]))
                hms.append(hm)
        '''

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        camera_params = {}

        camera_params['R'] = data['camera']['R']
        camera_params['T'] = data['camera']['t']
        camera_params['camera'] = data['camera']['camera']

        capture_idx = data['image']['capture']
        frame_idx = data['image']['frame_idx']
        seq_name = data['image']['seq_name']

        '''
        hand_dict = {}
        for hand_type in ['left', 'right']:
            hms = []
            for hIdx in range(7):
                hm = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(idx, hIdx, hand_type)))
                hm = cv.resize(hm, (img.shape[1], img.shape[0]))
                hms.append(hm)

            params = data['mano_params'][hand_type]
            handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                      torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())
            handV = handV[0].numpy()
            handJ = handJ[0].numpy()
            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]
        '''

        split_path = os.path.join(self.mano_dataset_folder, self.split)
        
        mano_data = {'right': {}, 'left': {}}

        for side in ['right', 'left']:
            for field_name, input_helper in self.input_helpers.items():
                try:
                    model = '%s_%s_%s' % (capture_idx, frame_idx, side) # !!! TO BE MODIFIED
                    field_data = input_helper.load(split_path, model)
                except Exception:
                    if self.no_except:
                        logger.warn(
                            'Error occured when loading field %s of model %s'
                            % (field_name.__class__.__name__, model)
                            # % (self.input_helper.__class__.__name__, model)
                        )
                        return None
                    else:
                        raise

                if isinstance(field_data, dict):
                    for k, v in field_data.items():
                        if k is None:
                            mano_data[side][field_name] = v
                        elif field_name == 'inputs':
                            mano_data[side][k] = v
                        else:
                            mano_data[side]['%s.%s' % (field_name, k)] = v
                else:
                    mano_data[side][field_name] = field_data

        if self.transforms is not None:
            for side in ['right', 'left']: 
                for tran_name, tran in self.transforms.items():
                    mano_data[side] = tran(mano_data[side])
                # if field_name in self.transform:
                #     data[field_name] = self.transform[field_name](data[field_name])

        if self.return_idx:
            mano_data[side]['idx'] = idx

        camera_params['right_root_xyz'] = mano_data['right']['root_xyz']
        camera_params['left_root_xyz'] = mano_data['left']['root_xyz']
        
        '''
        for side in ['right', 'left']:
            mano_params = self.annot_file[str(capture_idx)][str(frame_idx)][side]
            pose_para = torch.from_numpy(np.asarray(mano_params["pose"])).unsqueeze(0).float()
            shape = torch.from_numpy(np.asarray(mano_params["shape"])).unsqueeze(0).float()
            trans = torch.from_numpy(np.asarray(mano_params["trans"])).unsqueeze(0).float()

            if side == 'right':
                hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = self.right_mano_layer(pose_para, shape, trans, no_root_rot=False)
                right_root_xyz = hand_joints[0][0]
                camera_params['right_root_xyz'] = right_root_xyz
            elif side == 'left':
                hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = self.left_mano_layer(pose_para, shape, trans, no_root_rot=False)
                left_root_xyz = hand_joints[0][0]
                camera_params['left_root_xyz'] = left_root_xyz
        '''

        return imgTensor, maskTensor, denseTensor, camera_params, mano_data, idx
        
class SampleHandDataset(data.Dataset):
    ''' Sample Hands dataset class.
    '''

    def __init__(self, dataset_folder, input_helpers, split=None,
                 no_except=True, transforms=None, return_idx=False):
        ''' Initialization of the the 3D articulated hand dataset.
        Args:
            dataset_folder (str): dataset folder
            input_helpers dict[(callable)]: helpers for data loading
            split (str): which split is used
            no_except (bool): no exception
            transform dict{(callable)}: transformation applied to data points
            return_idx (bool): wether to return index
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.input_helpers = input_helpers
        self.split = split
        self.no_except = no_except
        self.transforms = transforms
        self.return_idx = return_idx

        ## Get all models
        split_file = os.path.join(dataset_folder, split, 'datalist.txt')
        with open(split_file, 'r') as f:
            self.models = f.read().strip().split('\n')

        # # Read metadata file
        # metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        # if os.path.exists(metadata_file):
        #     with open(metadata_file, 'r') as f:
        #         self.metadata = yaml.load(f)
        # else:
        #     self.metadata = {
        #         c: {'id': c, 'name': 'n/a'} for c in categories
        #     } 
        
        # # Set index
        # for c_idx, c in enumerate(categories):
        #     self.metadata[c]['idx'] = c_idx

        # # Get all models
        # self.models = []
        # for c_idx, c in enumerate(categories):
        #     subpath = os.path.join(dataset_folder, c)
        #     if not os.path.isdir(subpath):
        #         logger.warning('Category %s does not exist in dataset.' % c)

        #     split_file = os.path.join(subpath, split + '.lst')
        #     with open(split_file, 'r') as f:
        #         models_c = f.read().split('\n')
            
        #     self.models += [
        #         {'category': c, 'model': m}
        #         for m in models_c
        #     ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        #print(self.subset)
        #print(len(self.models))
        return len(self.models) // self.subset

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        # category = self.models[idx]['category']
        # model = self.models[idx]['model']
        # c_idx = self.metadata[category]['idx']
        model = self.models[idx]
        #print(model)

        split_path = os.path.join(self.dataset_folder, self.split)
        
        data = {}

        for field_name, input_helper in self.input_helpers.items():
            try:
                field_data = input_helper.load(split_path, model, idx)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name.__class__.__name__, model)
                        # % (self.input_helper.__class__.__name__, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    elif field_name == 'inputs':
                        data[k] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        # for field_name, field in self.fields.items():
        #     try:
        #         field_data = field.load(model_path, idx, c_idx)
        #     except Exception:
        #         if self.no_except:
        #             logger.warn(
        #                 'Error occured when loading field %s of model %s'
        #                 % (field_name, model)
        #             )
        #             return None
        #         else:
        #             raise

        #     if isinstance(field_data, dict):
        #         for k, v in field_data.items():
        #             if k is None:
        #                 data[field_name] = v
        #             else:
        #                 data['%s.%s' % (field_name, k)] = v
        #     else:
        #         data[field_name] = field_data

        # if self.transform is not None:
        #     data = self.transform(data)

        if self.transforms is not None:
            
            for tran_name, tran in self.transforms.items():
                data = tran(data)
                # if field_name in self.transform:
                #     data[field_name] = self.transform[field_name](data[field_name])

        if self.return_idx:
            data['idx'] = idx
        
        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.
        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True
