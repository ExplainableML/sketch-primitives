import torch.nn as nn
from ..transformer import StrokeEncoderTransformer
from .networks import VGG_Network, InceptionV3_Network, Resnet50_Network
from torch import optim
import torch
import numpy as np
import time
import torch.nn.functional as F


class FGSBIR_Model(nn.Module):
    def __init__(self, backbone, Nz=256, loss_type='cross_entropy', ce_temp=0.2):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(backbone + '_Network(scale=Nz//2)')

        self.loss_type = loss_type

        if self.loss_type == 'cross_entropy':
            temperature = torch.tensor([ce_temp])
            self.register_buffer('temperature', temperature)
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss = nn.TripletMarginLoss(margin=0.2)

    def forward(self, x_raw, n_strokes, scale, translate, seq_mask,
                positive_img, negative_img, sketch_img, positive_name,
                mask=None, **kwargs):

        positive_feature = self.sample_embedding_network(positive_img)
        sketch_feature = self.sample_embedding_network(sketch_img, finetune=True)

        if self.loss_type == 'cross_entropy':
            labels = torch.arange(sketch_feature.shape[0]).to(sketch_feature.device)

            positive_name = np.array(positive_name)
            mask = [pn == positive_name for pn in positive_name]
            mask = np.stack(mask)
            np.fill_diagonal(mask, False)
            mask_bool = torch.tensor(mask)
            mask = torch.zeros_like(mask_bool, dtype=torch.float)
            mask = mask.masked_fill(mask_bool, -np.inf).to(sketch_feature.device)

            similarity_matrix = torch.matmul(sketch_feature, positive_feature.T) / self.temperature
            loss = (self.criterion(similarity_matrix+mask, labels) + self.criterion(similarity_matrix.T+mask, labels)) / 2.
        else:
            negative_feature = self.sample_embedding_network(negative_img)
            loss = self.loss(sketch_feature, positive_feature, negative_feature)

        output = {'loss': loss,
                  'pcoords_raw': x_raw,
                  'pcoords_n_strokes': n_strokes,
                  'pcoords_scale': scale,
                  'pcoords_translate': translate,
                  'pcoords_seq_mask': seq_mask,
                  'sketch_feature': sketch_feature.detach(),
                  'image_feature': positive_feature.detach()}

        return output
