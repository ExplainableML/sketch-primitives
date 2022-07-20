import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from .transformer import StrokeEncoderTransformer

class SketchClassifier(nn.Module):
    def __init__(self, n_classes, Nz=256, use_pos_embed=False,
                 use_sinusoid_embed=False, stroke_len=25):
        super().__init__()
        self.Nz = Nz
        self.encoder = StrokeEncoderTransformer(
            Nz=Nz, embed_sketch=True, n_layer_strokes=3, use_pos_embed=use_pos_embed,
            use_sinusoid_embed=use_sinusoid_embed, out_dim=n_classes, stroke_len=stroke_len
        )
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x_raw, label, n_strokes, scale, translate, stroke_lens, seq_mask, position, **kwargs):
        # encode:
        cls_logits, _ = self.encoder(x_raw.transpose(0, 1), n_strokes, scale, translate, stroke_lens, seq_mask, position)
        loss = self.criterion(cls_logits, label)

        _, pred_main = torch.max(cls_logits, 1)
        acc = (label == pred_main).float()
        clspred_main_prob = torch.softmax(cls_logits, dim=1).gather(1, pred_main.unsqueeze(1)).squeeze(1)

        output = {'loss': loss,
                  'acc_main': acc,
                  'cls_pred_main': pred_main,
                  'cls_pred_main_prob': clspred_main_prob,
                  'pcoords_raw': x_raw,
                  'pcoords_n_strokes': n_strokes,
                  'pcoords_scale': scale,
                  'pcoords_translate': translate,
                  'pcoords_seq_mask': seq_mask}

        return output
