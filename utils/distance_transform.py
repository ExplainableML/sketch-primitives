import torch
import torch.nn.functional as F

def compute_distance_transform_l2(batch, prims, grid_sqz, batch_mask=None,
                                  prims_mask=None, gamma=3.):
    gt_dist = distance_transform(batch, grid_sqz, mask=batch_mask, gamma=gamma)
    if prims.dim() == 4:
        dims = prims.shape
        prims = prims.view(-1, prims.size(2), prims.size(3))
    else:
        dims = None
    prim_dist = distance_transform(prims, grid_sqz, mask=prims_mask, gamma=gamma)
    if dims is not None:
        prim_dist = prim_dist.view(dims[0], dims[1], prim_dist.size(-2), prim_dist.size(-1))
        gt_dist = gt_dist.unsqueeze(1).expand(prim_dist.size())
    return F.mse_loss(gt_dist, prim_dist, reduction='none').mean([-2, -1])

def make_coordinate_grid(dim, scale=1.5):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = (dim, dim)
    x = torch.arange(w).float()
    y = torch.arange(h).float()

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = (torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2).unsqueeze(0).unsqueeze(0) / (h - 1) * 2) - 1
    meshed *= scale

    return meshed

def distance_transform(kps, grid_sqz, gamma=3., glb=False, mask=None):
    grid_size = grid_sqz.shape[-2]
    grid_sqz = grid_sqz.repeat(kps.size(0),1, 1, 1, 1)
    # kps = torch.index_select(kps_, 2, torch.LongTensor([1, 0]).to(kps_.device))
    pi_set = kps[:, :-1].unsqueeze(-2).unsqueeze(-2)
    pj_set = kps[:, 1:].unsqueeze(-2).unsqueeze(-2)

    # Compute r
    v_set = (pi_set - pj_set).repeat(1, 1, grid_size, grid_size, 1)
    v_norm = (v_set.pow(2)).sum(-1).unsqueeze(-1)
    u_set = (grid_sqz - pj_set)

    uv = torch.bmm(u_set.view(-1, 1, 2), (v_set).view(-1, 2, 1)).view(kps.shape[0], -1, grid_size, grid_size, 1)
    rs = torch.clamp(uv / v_norm, 0, 1)#.detach()
    rs.masked_scatter_(rs.isnan(), uv)
    #rs = rs.detach()

    betas = ((u_set - rs * v_set).pow(2)).sum(-1)
    betas = torch.exp(-gamma * betas)

    if mask is not None:
        betas = betas * (~mask[:, 1:]).float().unsqueeze(-1).unsqueeze(-1)

    betas = betas.max(1)[0]
    if glb:
        betas = betas.max(0)[0]

    return betas
