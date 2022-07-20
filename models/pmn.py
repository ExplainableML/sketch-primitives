import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .transformer import StrokeEncoderTransformer

from utils.coords_utils import resample_stroke, normalize_strokes
from utils.distance_transform import compute_distance_transform_l2, make_coordinate_grid
from utils.transformations import ToTransformMatrix, get_predefined_primitives, transform_primitives


class PrimitiveMatchingNetwork(nn.Module):
    def __init__(self, Nz=256,
                 temperature=0.2,
                 primitive_names=['square', 'triangle', 'circle', 'line', 'halfcircle', 'u', 'corner', 'dot'],
                 n_stroke_points=25,
                 dt_gamma=6., use_pos_embed=False,
                 use_sinusoid_embed=False,
                 transform_bound=None, use_proportion_scale=False,
                 learn_compatibility=False, transformations='rsr'):
        super().__init__()
        self.Nz = Nz
        self.encoder = StrokeEncoderTransformer(
            Nz=Nz, embed_sketch=False, n_layer_strokes=6,
            use_pos_embed=use_pos_embed, use_sinusoid_embed=use_sinusoid_embed
        )

        #ALL_TRANSFORMATIONS = ['translate', 'rotate', 'scale_full_bounded', 'scale_full', 'uni_scale', 'proportion_scale', 'shear_x', 'shear_y'][::-1]

        transforms = []
        self.parameters_per_transform = []
        for ts in transformations:
            if ts == 'r':
                transforms.append('rotate')
                self.parameters_per_transform.append(1)
            elif ts == 's':
                if use_proportion_scale:
                    transforms.append('proportion_scale')
                    self.parameters_per_transform.append(1)
                else:
                    transforms.append('scale_full')
                    self.parameters_per_transform.append(2)
            elif ts == 'h':
                transforms.append('shear_x')
                transforms.append('shear_y')
                self.parameters_per_transform.append(1)
                self.parameters_per_transform.append(1)

        self.transformations = transforms
        self.transformation_layers = nn.ModuleList([ToTransformMatrix(t, bound=transform_bound) for t in self.transformations])

        self.learn_compatibility = learn_compatibility
        self.n_stroke_points = n_stroke_points

        self.init_primitives(primitive_names)

        primnet_in = 2 * Nz
        if self.learn_compatibility:
            primnet_in = primnet_in // 2
        self.primnet = nn.Sequential(
                nn.Linear(primnet_in, primnet_in//2),
                nn.BatchNorm1d(primnet_in//2),
                nn.ReLU(inplace=True),
                nn.Linear(primnet_in//2, primnet_in//4),
                nn.BatchNorm1d(primnet_in//4),
                nn.ReLU(inplace=True),
                nn.Linear(primnet_in//4,
                          sum(self.parameters_per_transform))
        )

        temperature = torch.tensor([temperature])
        self.register_buffer('temperature', temperature)

        self.dt_gamma = dt_gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.params_criterion = nn.MSELoss(reduction='none')

        self.init_distance_transform()

        self.cache_labels_feats = None
        self.cache_tgt_labels = None
        self.cache_tgt_masks = None

    def init_primitives(self, primitive_names):
        self.primitive_names = primitive_names
        primitives = get_predefined_primitives(
            primitive_names, n_points=self.n_stroke_points
        )
        self.register_buffer('primitives', primitives)
        self.num_primitives = len(primitive_names)

        prim_is_dot = torch.tensor([torch.unique(pt).nelement() == 1 for pt in self.primitives])
        self.register_buffer('prim_is_dot', prim_is_dot.unsqueeze(0).unsqueeze(-1))
        self.prim_has_dot = self.prim_is_dot.any()
        if self.prim_has_dot:
            self.prim_dot_id = self.prim_is_dot[0, :, 0].nonzero(as_tuple=False).item()
        else:
            self.prim_dot_id = None

        primitives_seq_mask = torch.zeros(self.primitives.shape[:2], dtype=torch.bool)
        self.register_buffer('primitives_seq_mask', primitives_seq_mask)
        primitives_position = torch.linspace(-1., 1., steps=self.primitives.shape[1]).unsqueeze(0).repeat(self.primitives.shape[0], 1)
        self.register_buffer('primitives_position', primitives_position)

    def init_distance_transform(self):
        self.grid_size = 32
        coordinate_grid = make_coordinate_grid(self.grid_size)
        grid_sqz = coordinate_grid.repeat(1, self.n_stroke_points-1, 1, 1, 1)
        self.register_buffer('grid_sqz', grid_sqz)

    def calc_min_l2_dist(self, recon, target, resample=False, roll=False, invert=False):
        l2_dist = torch.linalg.norm(recon - target, dim=-1).mean(-1)
        min_l2_dist = l2_dist

        if resample:
            recon = torch.stack([resample_stroke(s.cpu(), n=self.n_stroke_points)[..., :2].to(s) for s in recon])
            l2_dist_res = torch.linalg.norm(recon - target, dim=-1).mean(-1)
            #min_l2_dist = torch.stack([min_l2_dist, l2_dist_res], dim=-1).min(-1)[0]
            min_l2_dist = l2_dist_res

        if invert:
            l2_dist_inv = torch.linalg.norm(recon.flip(dims=(1,)) - target, dim=-1).mean(-1)
            min_l2_dist = torch.stack([min_l2_dist, l2_dist_inv], dim=-1).min(-1)[0]

        if roll:
            best_l2_dist_roll = None
            for i in range(1, recon.shape[1]):
                recon_roll = recon.roll(i, dims=(1,))
                l2_dist_roll = torch.linalg.norm(recon_roll - target, dim=-1).mean(-1)
                if best_l2_dist_roll is None:
                    best_l2_dist_roll = l2_dist_roll
                else:
                    best_l2_dist_roll = torch.stack([best_l2_dist_roll, l2_dist_roll], dim=-1).min(-1)[0]
                if invert:
                    l2_dist_roll_inv = torch.linalg.norm(recon_roll.flip(dims=(1,)) - target, dim=-1).mean(-1)
                    best_l2_dist_roll = torch.stack([best_l2_dist_roll, l2_dist_roll_inv], dim=-1).min(-1)[0]

            min_l2_dist = torch.stack([min_l2_dist, best_l2_dist_roll], dim=-1).min(-1)[0]

        out = {'l2_dist_min': min_l2_dist, 'l2_dist_base': l2_dist}
        if resample:
            out['l2_dist_resample'] = l2_dist_res

        if invert:
            out['l2_dist_invert'] = l2_dist_inv

        if roll:
            out['l2_dist_roll'] = best_l2_dist_roll

        return out

    def forward(self, x_raw, x_raw_aug=None,
                n_strokes=None, scale=None, translate=None,
                seq_mask=None, position=None, n_tgt=None, main_mask=None,
                **kwargs):

        batch = x_raw
        target = x_raw_aug

        n_tgt = max(batch.shape[0], self.primitives.shape[0])
        if self.primitives.shape[0] > batch.shape[0]:
            print('WARNING: More primitives than batch strokes. Untested scenario.')
        target = self.primitives
        tgt_seq_mask = self.primitives_seq_mask
        tgt_position = self.primitives_position

        # encode:
        z, z_params = self.encoder(batch.transpose(0, 1), seq_mask=seq_mask, position=position)[0].split([self.Nz//2, self.Nz//2], dim=-1)
        zt, zt_params = self.encoder(target.transpose(0, 1), seq_mask=tgt_seq_mask, position=tgt_position)[0].split([self.Nz//2, self.Nz//2], dim=-1)

        if self.learn_compatibility:
            output = self.info_nce_loss(z, zt, n_tgt, main_mask, ignore_loss=((not self.training) or self.learn_compatibility))
        if self.learn_compatibility:
            prim_logits = output['pred_main_logits']

        output = {}
        output['loss'] = 0.

        if not self.learn_compatibility:
            z_params = torch.cat([z, z_params], dim=-1)
            zt_params = torch.cat([zt, zt_params], dim=-1)

        z_params = z_params.unsqueeze(1)
        zt_params = zt_params.unsqueeze(0)
        z_params = z_params.expand(z_params.shape[0], zt_params.shape[1], z_params.shape[2])
        zt_params = zt_params.expand(z_params.shape[0], zt_params.shape[1], zt_params.shape[2])
        z_params = z_params.reshape(-1, z_params.shape[-1])
        zt_params = zt_params.reshape(-1, zt_params.shape[-1])

        primnet_in = torch.cat([z_params, zt_params], dim=-1)

        prims_params = self.primnet(primnet_in)

        prims_params = prims_params.split(self.parameters_per_transform, dim=-1)
        prim_params = []
        for p, s, t in zip(prims_params, self.parameters_per_transform, self.transformations):
            pt = p.view(-1, self.num_primitives, s)
            if t == 'proportion_scale':
                pt = torch.cat([pt, -pt], dim=-1)
            if self.prim_has_dot:
                pt = pt.masked_fill(self.prim_is_dot, 0.)
            prim_params.append(pt)

        prims = transform_primitives(
            target, self.transformation_layers, prim_params
            #target, self.transformation_layers_noop, parameters_aug.split([2, 2], dim=-1)
        )

        prims = prims.squeeze(-3).squeeze(-2)[..., :2]
        prims = normalize_strokes(prims)[0]

        if self.training or not self.learn_compatibility:
            dt_loss = 10. * compute_distance_transform_l2(batch, prims, self.grid_sqz, batch_mask=seq_mask, gamma=self.dt_gamma)
        else:
            dt_loss = 0.
        if self.learn_compatibility:
            if self.training:
                if self.prim_has_dot:
                    loss_mask = dt_loss[:, self.prim_dot_id].isclose(torch.tensor([0.], device=dt_loss.device)).unsqueeze(1)
                    dt_loss = dt_loss.masked_fill(loss_mask, 0.)

                dt_loss = (dt_loss * torch.softmax(prim_logits, dim=1)).sum(1)
            prim_ids = prim_logits.argmax(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            if self.prim_has_dot:
                if self.training:
                    prim_ids[loss_mask] = self.prim_dot_id
                else:
                    dot_mask = batch.isclose(torch.tensor([0.], device=batch.device)).sum([-2, -1]) == (batch.shape[-2]*batch.shape[-1])
                    prim_ids[dot_mask] = self.prim_dot_id

        else:
            prim_ids = dt_loss.argmin(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            if self.prim_has_dot:
                loss_mask = dt_loss[:, self.prim_dot_id].isclose(torch.tensor([0.], device=dt_loss.device)).unsqueeze(1)
                dt_loss = dt_loss.masked_fill(loss_mask, 0.)
            dt_loss = dt_loss.mean(1)

        prims = prims.gather(1, prim_ids.expand(prims.size(0), 1, prims.size(2), prims.size(3)))
        prims = prims.squeeze(1)

        output['pcoords_prim_id'] = prim_ids.squeeze(-1)
        output['loss'] += dt_loss

        output['pcoords_raw'] = prims
        output['pcoords_n_strokes'] = n_strokes
        output['pcoords_scale'] = scale
        output['pcoords_translate'] = translate
        output['pcoords_seq_mask'] = self.primitives_seq_mask[:1].expand(prims.shape[:2])
        output['pcoords_position'] = self.primitives_position[:1].expand(prims.shape[:2])

        return output

    def info_nce_loss(self, features, targets, n_tgt=None, main_mask=None,
                      ignore_loss=False):

        if n_tgt is None:
            n_tgt = targets.shape[0]
        labels = torch.arange(n_tgt).to(features.device)

        features = F.normalize(features, dim=-1)
        targets = F.normalize(targets, dim=-1)

        similarity_matrix = torch.matmul(features, targets.T) / self.temperature
        main_simmat = similarity_matrix[:n_tgt]
        if main_mask is not None:
            main_simmat += main_mask
        mod_simmat = similarity_matrix.T[:n_tgt]
        if not ignore_loss:
            loss_main = self.criterion(main_simmat, labels)
            loss_mod = self.criterion(mod_simmat, labels)
            total_loss = (loss_main + loss_mod) / 2
        else:
            total_loss = 0.

        labels_mod = labels[:mod_simmat.shape[0]]

        _, pred_main = torch.max(main_simmat, 1)
        acc_main = (labels[:pred_main.shape[0]] == pred_main).float()
        _, pred_mod = torch.max(mod_simmat, 1)
        acc_mod = (labels_mod[:pred_mod.shape[0]] == pred_mod).float()

        output = {'loss': total_loss,
                  'acc_main': acc_main,
                  'acc_mod': acc_mod,
                  'pred_main': pred_main,
                  'pred_mod': pred_mod,
                  'pred_main_logits': main_simmat}

        return output

    def l2_eval(self, batch):
        prims = self.draw(batch)
        l2_dist = self.calc_min_l2_dist(prims, batch, resample=True, roll=True, invert=True)
        dist_transform = compute_distance_transform_l2(prims, batch, self.grid_sqz, gamma=self.dt_gamma)
        return l2_dist['l2_dist_min'], dist_transform

    def draw(self, **inputs):
        return self.forward(**inputs)

    def compute_l2(self,prims,batch):
        return self.calc_min_l2_dist(prims, batch, resample=True, roll=True, invert=True)
