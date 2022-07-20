import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.coords_utils import normalize_strokes

ALLOWED_TRANSFORMATIONS = ['identity', 'translate_x', 'translate_y', 'translate', 'scale', 'scale_full_bounded', 'scale_full', 'uni_scale', 'proportion_scale', 'shear_x', 'shear_y', 'rotate']


class CosSin(nn.Module):
    def __init__(self, dim):
        super(CosSin, self).__init__()
        self.dim = dim

    def forward(self, input):
        return torch.cat([torch.cos(input), torch.sin(input)], dim=self.dim)


class ToTransformMatrix(nn.Module):
    def __init__(self, transformation, no_activation=False, bound=None):
        super(ToTransformMatrix, self).__init__()

        assert transformation in ALLOWED_TRANSFORMATIONS


        if transformation == 'identity':
            mul = [
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            ]
            bias = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]

            def identity(x):
                return x
            activation = identity

        elif transformation == 'translate':
            mul = [
                [[0, 0, 1],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[0, 0, 0],
                 [0, 0, 1],
                 [0, 0, 0]]
            ]
            bias = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            #activation = nn.Tanh()
            def translate_act(x):
                return torch.tanh(x)# * 0.9
            activation = translate_act

        elif transformation == 'translate_x':
            mul = [
                [[0, 0, 1],
                 [0, 0, 0],
                 [0, 0, 0]]
            ]
            bias = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            #activation = nn.Tanh()
            def translate_act(x):
                return torch.tanh(x)# * 0.9
            activation = translate_act

        elif transformation == 'translate_y':
            mul = [
                [[0, 0, 0],
                 [0, 0, 1],
                 [0, 0, 0]]
            ]
            bias = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            #activation = nn.Tanh()
            def translate_act(x):
                return torch.tanh(x)# * 0.9
            activation = translate_act

        elif transformation == 'scale_full':
            mul = [
                [[1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ]

            bias = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]]

            activation = torch.exp

        elif transformation == 'scale_full_bounded':
            mul = [
                [[1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ]

            bias = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]]

            def scale_act(x):
                return torch.exp(x) + (bound if bound is not None else 0.)
            activation = scale_act

        elif transformation == 'scale':
            mul = [
                [[1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ]

            bias = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]]

            def scale_act(x):
                return 2. * torch.sigmoid(x) + 0.1
            activation = scale_act

        elif transformation == 'uni_scale':
            mul = [
                [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ]

            bias = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]]

            def scale_act(x):
                return 2. * torch.sigmoid(x) + 0.02
            activation = scale_act

        elif transformation == 'proportion_scale':
            mul = [
                [[-1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[0, 0, 0],
                 [0, -1, 0],
                 [0, 0, 0]]
            ]

            bias = [[1. + (bound if bound is not None else 0.02), 0, 0],
                    [0, 1. + (bound if bound is not None else 0.02) , 0],
                    [0, 0, 1]]

            def scale_act(x):
                return F.relu(torch.tanh(x))
            activation = scale_act

        elif transformation == 'shear_x':
            mul = [
                [[0, 1, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            ]
            bias = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            activation = nn.Tanh()

        elif transformation == 'shear_y':
            mul = [
                [[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 0]]
            ]
            bias = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            activation = nn.Tanh()

        elif transformation == 'rotate':
            mul = [
                [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]],

                [[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 0]]
            ]
            bias = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]]
            activation = CosSin(dim=2)

        self.mul = nn.Parameter(torch.tensor(mul).float().unsqueeze(0).unsqueeze(1),
                                requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(bias).float().unsqueeze(0).unsqueeze(1),
                                 requires_grad=False)
        if no_activation:
            self.activation = None
        else:
            self.activation = activation

    def forward(self, input):
        if not (self.activation is None):
            input = self.activation(input)
        return (input.unsqueeze(3).unsqueeze(4) * self.mul).sum(2) + self.bias


def add_edge_points(points, n_points):
    points = np.array(points)
    new_points = []
    pnts_per_egdge = (n_points-1) // (len(points)-1)
    extra_points = (n_points-1) % (len(points)-1)
    for i in range(len(points)-1):
        if extra_points > 0:
            extra_points -= 1
            ep = 1
        else:
            ep = 0
        edge_points = np.linspace(points[i], points[i+1], pnts_per_egdge+ep, endpoint=False)
        new_points.append(edge_points)
    new_points.append(points[-1][np.newaxis])
    new_points = np.concatenate(new_points)
    return new_points


def create_circle(n_points, scale=0.5, percent=1.):
    x = np.linspace(0., 2*np.pi*percent, n_points)
    points = np.stack([np.sin(x), np.cos(x)], axis=1)
    points = points * scale
    return points


def get_predefined_primitives(primitive_names, n_points=25, normalize=True,
                              scale=1.):
    points = []

    for pname in primitive_names:
        # Square
        if pname == 'square':
            pnts = [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]
            pnts = add_edge_points(pnts, n_points)
            points.append(torch.from_numpy(pnts).float())

        # Triangle
        elif pname == 'triangle':
            pnts = [[-0.5, 0.44], [0.5, 0.44], [0., -0.44], [-0.5, 0.44]]
            pnts = add_edge_points(pnts, n_points)
            points.append(torch.from_numpy(pnts).float())

        # Circle
        elif pname == 'circle':
            pnts = create_circle(n_points)
            points.append(torch.from_numpy(pnts).float())

        # Half-circle
        elif pname == 'halfcircle':
            pnts = create_circle(n_points, percent=0.5)
            points.append(torch.from_numpy(pnts).float())

        # Line
        elif pname == 'line':
            pnts = [[-0.5, 0.], [0.5, 0.]]
            pnts = add_edge_points(pnts, n_points)
            points.append(torch.from_numpy(pnts).float())

        # Corner
        elif pname == 'corner':
            pnts = [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
            pnts = add_edge_points(pnts, n_points)
            points.append(torch.from_numpy(pnts).float())

        # u
        elif pname == 'u':
            pnts = [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]
            pnts = add_edge_points(pnts, n_points)
            points.append(torch.from_numpy(pnts).float())

        # dot (single point
        elif pname == 'dot':
            pnts = [[0., 0.]]
            pnts = np.array(pnts).repeat(n_points, axis=0)
            points.append(torch.from_numpy(pnts).float())

    primitives = torch.stack(points)
    if normalize:
        primitives, _, _ = normalize_strokes(primitives)

    if scale != 1.:
        primitives = primitives * scale

    return primitives

def compute_full_transformation(transform_layers, transformation_parameters):
    mat = None
    for t_layer, params in zip(transform_layers, transformation_parameters):
        t_mat = t_layer(params)

        if mat is None:
            mat = t_mat
        else:
            mat = t_mat @ mat

    return mat


def transform_primitives(primitives, transform_layers, transforms_params, prim_ids=None):
    full_transforms = compute_full_transformation(transform_layers, transforms_params)

    if prim_ids is None:
        if primitives.size(0) != transforms_params[0].size(0):
            full_prim_htan = primitives.unsqueeze(0).expand([transforms_params[0].size(0)]+list(primitives.size()))
        else:
            full_prim_htan = primitives.unsqueeze(2)
    else:
        full_prim_htan = primitives.unsqueeze(0).repeat(prim_ids.size(0), 1, 1, 1).gather(1, prim_ids.repeat(1, 1, primitives.size(1), primitives.size(2)))

    full_prim_htan = torch.cat(
        [full_prim_htan, torch.ones(full_prim_htan.size()[:-1]).unsqueeze(-1).to(full_prim_htan.device)], dim=-1)

    full_prim_htan = full_prim_htan.unsqueeze(-2) @ full_transforms.unsqueeze(-3).transpose(-2, -1)

    return full_prim_htan
