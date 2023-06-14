import sys
import torch
import torch.nn as nn
from torch import distributions as dist

from dependencies.intaghand.models.encoder import ResNetSimple 
from dependencies.intaghand.models.model_attn.img_attn import *  
from dependencies.intaghand.models.model_attn.self_attn import *  
from dependencies.intaghand.models.model_attn.gcn import GraphLayer  

def hand_joint_graph(v_num=21):
    '''
        connected by only root
    '''
    graph = torch.zeros((v_num,v_num))
    edges = torch.tensor([[0,13],
                            [13,14],
                            [14,15],
                            [15,16],
                            [0,1],
                            [1,2],
                            [2,3],
                            [3,17],
                            [0,4],
                            [4,5],
                            [5,6],
                            [6,18],
                            [0,10],
                            [10,11],
                            [11,12],
                            [12,19],
                            [0,7],
                            [7,8],
                            [8,9],
                            [9,20]])
    graph[edges[:,0], edges[:,1]] = 1.0

    return graph


class DualGraphLayer(nn.Module):
    def __init__(self,
                 verts_in_dim=48,
                 verts_in_dim_2=48,
                 verts_out_dim=16,
                 graph_L_Left=None,
                 graph_L_Right=None,
                 graph_k=2,
                 graph_layer_num=3,
                 img_size=64,
                 img_f_dim=64,
                 grid_size=8,
                 grid_f_dim=64,
                 n_heads=4,
                 dropout=0,
                 is_inter_attn=True
                 ):
        super().__init__()

        self.verts_num = graph_L_Left.shape[0]
        self.verts_in_dim = verts_in_dim
        self.img_size = img_size
        self.img_f_dim = img_f_dim
        self.inter_attn = is_inter_attn

        self.graph_left = GraphLayer(verts_in_dim, verts_out_dim,
                                     graph_L_Left, graph_k, graph_layer_num,
                                     dropout)
        self.graph_right = GraphLayer(verts_in_dim, verts_out_dim,
                                     graph_L_Right, graph_k, graph_layer_num,
                                     dropout)
        self.img_ex_left = img_ex(img_size, img_f_dim,
                                  grid_size, grid_f_dim,
                                  verts_in_dim_2, 
                                  n_heads=n_heads,
                                  dropout=dropout)
        self.img_ex_right = img_ex(img_size, img_f_dim,
                                  grid_size, grid_f_dim,
                                  verts_in_dim_2, 
                                  n_heads=n_heads,
                                  dropout=dropout)        

    def forward(self, left_joint_ft, right_joint_ft, img_f):
        BS1, V, f = left_joint_ft.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim
        BS2, V, f = right_joint_ft.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim
        BS3, C, H, W = img_f.shape
        assert C == self.img_f_dim
        assert H == self.img_size
        assert W == self.img_size
        assert BS1 == BS2
        assert BS2 == BS3
        BS = BS1

        left_joint_ft_graph = self.graph_left(left_joint_ft)
        right_joint_ft_graph = self.graph_right(right_joint_ft)

        left_joint_ft = self.img_ex_left(img_f, torch.cat([left_joint_ft, left_joint_ft_graph], 2))
        right_joint_ft = self.img_ex_right(img_f, torch.cat([right_joint_ft, right_joint_ft_graph], 2))

        return left_joint_ft, right_joint_ft


class ArticulatedHandNetKptsRef(nn.Module):
    ''' Keypoint Refinement Network class.
    Args:
        device (device): torch device
    '''

    def __init__(self, device='cuda'):

        super().__init__()

        self._device = device

        verts_in_dim = [48, 64, 80, 96]
        verts_in_dim_2 = [64, 80, 96, 112]
        verts_out_dim = [16, 16, 16, 16]

        graph_L_Left = [
            hand_joint_graph().to(device),
        ] * 4
        graph_L_Right = [
            hand_joint_graph().to(device),
        ] * 4

        graph_k =[2, 2, 2, 2]
        graph_layer_num = [2, 2, 2, 2]
        img_size = [64, 64, 64, 64]
        img_f_dim = [256, 256, 256, 256]
        grid_size = [8, 8, 8, 8]
        grid_f_dim = [64, 82, 96, 112]
        n_heads = 4

        self.image_encoder = ResNetSimple(model_type='resnet50',
                                          pretrained=True,
                                          fmapDim=[128, 128, 128, 128],
                                          handNum=2,
                                          heatmapDim=21)

        self.img_proj_layer = nn.Conv2d(512, 256, 1)
        self.img_final_layer = nn.Conv2d(256, 32, 1)

        self.left_id_emb = nn.Embedding(21, 16)
        self.right_id_emb = nn.Embedding(21, 16)

        self.left_pt_emb = nn.Sequential(nn.Conv1d(3, 32, 1),
                                         nn.ReLU(),
                                         nn.Dropout(0.01),
                                         nn.Conv1d(32, 32, 1))
        self.right_pt_emb = nn.Sequential(nn.Conv1d(3, 32, 1),
                                          nn.ReLU(),
                                          nn.Dropout(0.01),
                                          nn.Conv1d(32, 32, 1))
        
        self.left_reg = nn.Sequential(nn.Conv1d(112, 32, 1),
                                      nn.ReLU(),
                                      nn.Dropout(0.01),
                                      nn.Conv1d(32, 3, 1))
        self.right_reg = nn.Sequential(nn.Conv1d(112, 32, 1),
                                       nn.ReLU(),
                                       nn.Dropout(0.01),
                                       nn.Conv1d(32, 3, 1))

        self._device = device

        self.gcn_layers = nn.ModuleList()
        
        for i in range(len(verts_in_dim)):
            self.gcn_layers.append(DualGraphLayer(verts_in_dim=verts_in_dim[i],
                                                  verts_in_dim_2=verts_in_dim_2[i],
                                                  verts_out_dim=verts_out_dim[i],
                                                  graph_L_Left=graph_L_Left[i].detach().cpu().numpy(),
                                                  graph_L_Right=graph_L_Right[i].detach().cpu().numpy(),
                                                  graph_k=graph_k[i],
                                                  graph_layer_num=graph_layer_num[i],
                                                  img_size=img_size[i],
                                                  img_f_dim=img_f_dim[i],
                                                  grid_size=grid_size[i],
                                                  grid_f_dim=grid_f_dim[i],
                                                  n_heads=n_heads,
                                                  dropout=0.01))


    def forward(self, img, camera_params, joints, **kwargs):
        ''' Performs a forward pass through the network.
        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        
        # 2D feature generate
        batch_size = img.shape[0]

        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.image_encoder(img.cuda())
        img_f, hms_f, dp_f = img_fmaps[-1], hms_fmaps[-1], dp_fmaps[-1] 

        # position embedding
        joint_ids = torch.arange(21, dtype=torch.long, device=self._device)
        joint_ids = joint_ids.unsqueeze(0).repeat(batch_size, 1)

        left_id_emb = self.left_id_emb(joint_ids)
        right_id_emb = self.right_id_emb(joint_ids)

        # point embedding
        left_pt_emb = self.left_pt_emb(joints['left'].transpose(1,2).float())
        left_joint_ft = torch.cat((left_id_emb, left_pt_emb.transpose(1,2)), 2) # [32, 21, 48]

        right_pt_emb = self.right_pt_emb(joints['right'].transpose(1,2).float())
        right_joint_ft = torch.cat((right_id_emb, right_pt_emb.transpose(1,2)), 2) # [32, 21, 48]

        img_ft = torch.cat((img_f, hms_f, dp_f), 1) 
        img_ft = self.img_proj_layer(img_ft) # [bs, 64, 64, 64]

        for i in range(len(self.gcn_layers)):
            left_joint_ft, right_joint_ft = self.gcn_layers[i](left_joint_ft, right_joint_ft, img_ft)

        left_joint_res = self.left_reg(left_joint_ft.transpose(1,2)) # 32, 3, 21
        right_joint_res = self.left_reg(right_joint_ft.transpose(1,2))

        '''
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(left_joints[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/left_joints_org.ply", pcd)
        pcd.points = o3d.utility.Vector3dVector(right_joints[0].cpu().detach().numpy())
        o3d.io.write_point_cloud("debug/right_joints_org.ply", pcd)
        '''
        
        #import pdb; pdb.set_trace()
        left_joints = left_joint_res.transpose(1,2)
        right_joints = right_joint_res.transpose(1,2)

        return left_joints, right_joints


    def to(self, device):
        ''' Puts the model to the device.
        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
