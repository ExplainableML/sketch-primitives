import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from .transformer import StrokeEncoderTransformer
from utils.plt_blit import BlitManager
from utils.coords_utils import make_image_fast

def calc_segments_per_sketch(segments, stroke_lens, n_strokes):
    if len(segments) == 1:
        segments = segments[0]
    elif not isinstance(segments, torch.Tensor):
        segments = torch.stack(segments).T.flatten()

    grouped_segments = []
    group = []
    cnt = 0
    sid = 0
    for seg in segments:
        group.append(seg)
        cnt += seg
        if cnt == stroke_lens[sid]:
            grouped_segments.append(torch.stack(group))
            cnt = 0
            sid += 1
            group = []

    grouped_segments_len = [gs.shape[0] for gs in grouped_segments]

    segments_per_sketch = []
    cnt = 0
    for n_s in n_strokes:
        segments_per_sketch.append(sum(grouped_segments_len[cnt:cnt+n_s]))
        cnt += n_s
    segments_per_sketch = torch.tensor(segments_per_sketch, device=n_strokes.device)

    return grouped_segments, grouped_segments_len, segments_per_sketch

class Communication(nn.Module):
    def __init__(self, classifier, Nz=256, comm_mode='human',
                 loss_type='class', budget=1.0):
        super().__init__()
        assert comm_mode in ['human', 'random']
        self.Nz = Nz
        self.classifier = classifier
        if self.classifier is not None:
            for p in self.classifier.parameters():
                p.requires_grad = False

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.comm_mode = comm_mode
        self.loss_type = loss_type
        self.budget = budget
        self.gamma = 0.9

        if self.loss_type == 'fgsbir' and not self.classifier.use_transformer:
            fig, ax = plt.subplots(figsize=(2.99, 2.99), constrained_layout=True)
            ax.set_axis_off()
            fig.add_axes(ax)
            self.blit_manager = BlitManager(ax, fig.canvas)
            fig.canvas.draw()

            self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])


    def forward(self, x_raw, label, n_strokes, scale, translate, segments=None, stroke_lens=None, seq_mask=None, position=None, **kwargs):

        grouped_segments, grouped_segments_len, segments_per_sketch = calc_segments_per_sketch(
            segments, stroke_lens,
            kwargs['n_strokes_orig'] if 'n_strokes_orig' in kwargs else n_strokes
        )

        # encode:
        ol_size = (n_strokes.shape[0], n_strokes.max().item())
        if self.comm_mode.endswith('human'):
            arange_size = ol_size[1]
            order_logits = -torch.arange(arange_size, dtype=x_raw.dtype, device=x_raw.device).unsqueeze(0).expand((ol_size[0], arange_size))
            order_logits = order_logits.view(ol_size)
        else: # 'random'
            order_logits = torch.randn(ol_size, device=x_raw.device)

        padding_mask = torch.zeros(ol_size, device=x_raw.device, dtype=torch.bool)
        stroke_id = 0
        for i, n_s in enumerate(n_strokes):
            pad_len = ol_size[1] - n_s
            if pad_len != 0:
                padding_mask[i, -pad_len:] = True
            stroke_id += n_s

        order_logits = order_logits.masked_fill(padding_mask, float('-inf'))
        order_all = torch.full_like(order_logits, float('-inf'))

        final_cls_logits = None
        all_cls_logits = []
        cls_input = torch.zeros_like(x_raw)
        mask = torch.ones_like(padding_mask)
        grad_neg_mask = torch.zeros_like(x_raw[:, 0, 0])
        points_mask = None
        losses = []

        budget_n_strokes = torch.ceil(segments_per_sketch * self.budget).int()
        budget_n_strokes = torch.stack([budget_n_strokes, n_strokes]).min(0)[0]
        n_iters = max(budget_n_strokes)

        for i in range(n_iters):
            done = order_logits.isinf().sum(1) == order_logits.shape[1]
            done = done.unsqueeze(1).expand(order_logits.shape)

            order_logits = order_logits.masked_fill(done, 0.)

            next_order = F.one_hot(order_logits.argmax(dim=1), num_classes=order_logits.shape[1])
            next_order = next_order.float()

            next_order_flat = []
            global_stroke_id = 0
            for bi, (no, n_s) in enumerate(zip(next_order, n_strokes)):
                next_order_flat.append(no[:n_s])
                global_stroke_id += n_s
            next_order_flat = torch.cat(next_order_flat)
            next_order_flat = next_order_flat.masked_fill(next_order_flat.isnan(), 0.)

            active_sketches = i < budget_n_strokes
            active_strokes = torch.cat([a.repeat(n_s) for a, n_s in zip(active_sketches, n_strokes)])

            grad_neg_mask += active_strokes * next_order_flat
            cls_input = x_raw * grad_neg_mask.unsqueeze(-1).unsqueeze(-1)
            points_mask_input = None
            mask = mask & ~(active_sketches.unsqueeze(1) & next_order.bool()).detach()
            if self.loss_type == 'fgsbir':
                if not self.classifier.use_transformer:
                    sketch_images= []
                    n_s = list(n_strokes)
                    # TODO: remove duplicate code
                    if points_mask_input is None:
                        for ci, s, t in zip(cls_input.split(n_s), scale.split(n_s), translate.split(n_s)):
                            sketch_img = make_image_fast(self.blit_manager, (ci.cpu(), s.cpu(), t.cpu(), None), no_axis=True, color='black', linewidth=1, enlarge=2.0, arbitrary_mask=True)
                            sketch_images.append(self.img_transform(torch.tensor(sketch_img, dtype=torch.float)/255.))
                    else:
                        for ci, s, t, m in zip(cls_input.split(n_s), scale.split(n_s), translate.split(n_s), points_mask_input[:, :-1].split(n_s)):
                            sketch_img = make_image_fast(self.blit_manager, (ci.cpu(), s.cpu(), t.cpu(), m.cpu()), no_axis=True, color='black', linewidth=1, enlarge=2.0, arbitrary_mask=True)
                            sketch_images.append(self.img_transform(torch.tensor(sketch_img, dtype=torch.float)/255.))
                    kwargs['sketch_img'] = torch.stack(sketch_images).to(cls_input)
                outputs = self.classifier(cls_input, n_strokes, scale, translate, seq_mask=seq_mask, position=position, stroke_lens=stroke_lens, mask=torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1), points_mask=points_mask_input, **kwargs)
                cls_loss = outputs['loss'][active_sketches]
                sketch_feature = outputs['sketch_feature']
                image_feature = outputs['image_feature']
            elif self.loss_type == 'class':
                cls_logits, _ = self.classifier(cls_input.transpose(0, 1), n_strokes, scale, translate, seq_mask=seq_mask, position=position, mask=torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1), points_mask=points_mask_input, **kwargs)

                all_cls_logits.append(cls_logits)
                if final_cls_logits is None:
                    final_cls_logits = cls_logits.clone().detach()
                else:
                    final_cls_logits[active_sketches] = cls_logits[active_sketches].clone().detach()

                cls_loss = self.criterion(cls_logits[active_sketches], label[active_sketches])
                cls_rank = (cls_logits.argsort(1, descending=True) == label.unsqueeze(1)).nonzero()[:, 1]
            elif self.loss_type == 'none':
                cls_loss = torch.tensor([0.])
            else:
                raise NotImplementedError

            losses.append(cls_loss)

            order_all += 1
            order_all = order_all.masked_fill(next_order.bool() & active_sketches.unsqueeze(1), 0.)

            order_logits = order_logits.masked_fill(next_order.bool(), float('-inf'))

        loss = torch.cat(losses)
        assert not loss.isnan().any()

        if self.loss_type == 'class':
            _, pred_main = torch.max(final_cls_logits, 1)
            acc = (label == pred_main).float()

            all_cls_logits = torch.stack(all_cls_logits, dim=1)
            _, all_pred_main = torch.max(all_cls_logits, 2)
            all_acc = (label.unsqueeze(1) == all_pred_main).float()
            clspred_main_prob = torch.softmax(all_cls_logits, dim=2).gather(2, all_pred_main.unsqueeze(2)).squeeze(2)

        x_raw_reordered = []
        scale_reordered = []
        translate_reordered = []
        order_ids_reordered = []
        n_s = n_strokes.cpu().numpy().tolist()
        xr = x_raw.split(n_s)
        sc = scale.split(n_s)
        tr = translate.split(n_s)
        pcoords_n_strokes = budget_n_strokes

        for ol, x, s, t, n_s in zip(order_all, xr, sc, tr, pcoords_n_strokes):
            assert n_s == (~ol.isinf()).sum()
            order = ol.argsort(dim=0, descending=True)
            order = order[:n_s]
            x_raw_reordered.append(x[order])
            scale_reordered.append(s[order])
            translate_reordered.append(t[order])
        x_raw_reordered = torch.cat(x_raw_reordered)
        scale_reordered = torch.cat(scale_reordered)
        translate_reordered = torch.cat(translate_reordered)

        if self.loss_type == 'class':
            output = {'loss': loss,
                      'acc_main': acc,
                      'acc_main_all': all_acc,
                      'cls_pred_main': all_pred_main,
                      'cls_pred_main_prob': clspred_main_prob,
                      'pcoords_raw': x_raw_reordered,
                      'pcoords_n_strokes': pcoords_n_strokes,
                      'pcoords_scale': scale_reordered,
                      'pcoords_translate': translate_reordered,
                      'pcoords_seq_mask': None,
                      'segments_per_sketch': segments_per_sketch,
            }
        elif self.loss_type == 'fgsbir':
            output = {'loss': loss,
                      'pcoords_raw': x_raw_reordered,
                      'pcoords_n_strokes': pcoords_n_strokes,
                      'pcoords_scale': scale_reordered,
                      'pcoords_translate': translate_reordered,
                      'pcoords_seq_mask': None,
                      'segments_per_sketch': segments_per_sketch,
                      'sketch_feature': sketch_feature.detach(),
                      'image_feature': image_feature.detach(),
            }
        else:
            output = {'loss': loss,
                      'pcoords_raw': x_raw_reordered,
                      'pcoords_n_strokes': pcoords_n_strokes,
                      'pcoords_scale': scale_reordered,
                      'pcoords_translate': translate_reordered,
                      'pcoords_seq_mask': None,
                      'segments_per_sketch': segments_per_sketch,
                      'stroke_ids': order_ids_reordered,
            }

        return output

    def draw(self, **inputs):
        return self.forward(**inputs)
