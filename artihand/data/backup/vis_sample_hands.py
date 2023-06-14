import os
import logging
from torch.utils import data
import numpy as np
import yaml


logger = logging.getLogger(__name__)

class VisSampleHandDataset(data.Dataset):
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

        #self.vis_idx = np.load('/data/interhand2.6m/test/transfer_vis/vis_list.npy')
        self.vis_idx = np.asarray(range(0, 270000, 10))

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
        return self.vis_idx.shape[0]
        #return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''

        idx = self.vis_idx[idx]

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
