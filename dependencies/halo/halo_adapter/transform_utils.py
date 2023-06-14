import torch


def xyz_to_xyz1(xyz):
    """ Convert xyz vectors from [BS, ..., 3] to [BS, ..., 4] for matrix multiplication
    """
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    # print("xyz shape", xyz.shape)
    # print("one", ones.shape)
    return torch.cat([xyz, ones], dim=-1)


def pad34_to_44(mat):
    last_row = torch.tensor([0., 0., 0., 1.], device=mat.device).reshape(1, 4).repeat(*mat.shape[:-2], 1, 1)
    return torch.cat([mat, last_row], dim=-2)