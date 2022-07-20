import os
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.metrics import TbLogger, Accumulator, log_metrics, log_images
from utils.coords_utils import PRIM_COLORS, PRIM_ORDER, make_image_fast
from utils.plt_blit import BlitManager


class Trainer:
    def __init__(self, model_type, model, train, learning_rate, epochs,
                 dataloader, test_dataloader, label_names, device, log_dir,
                 log_every, n_log_img, save_every, view_scale, weight_decay):
        self.model_type = model_type
        self.model = model
        self.train = train
        self.epochs = epochs if self.train else 1
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.label_names = label_names
        self.device = device
        self.log_dir = log_dir + ('' if self.train else '_test')
        self.log_every = log_every
        self.n_log_img = n_log_img
        self.save_every = save_every
        self.view_scale = view_scale
        self.best_result = None

        self.tblogger = TbLogger(self.log_dir)
        if self.train:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.metrics = defaultdict(Accumulator)
        self.update_steps = 0
        self.blit_manager = None

    def run(self):
        for e in range(self.epochs):
            if not self.train:
                self.test_dataloader.generator.manual_seed(42)
            tqdm_loader = tqdm(self.dataloader if self.train else self.test_dataloader)

            self.update_steps = self.run_epoch(tqdm_loader, self.train,
                                               self.update_steps)

            if self.train:
                self.test_dataloader.generator.manual_seed(42)
                self.run_epoch(tqdm(self.test_dataloader), False, 0)

        self.tblogger.close()

    def update_metrics(self, outputs, train):
        prefix = 'train' if train else 'test'

        if isinstance(outputs['loss'],float):
            self.metrics[f'{prefix}/loss'].add(value=outputs['loss'])
        else:
            self.metrics[f'{prefix}/loss'].add(value=outputs['loss'].mean().item())

        if 'acc_main' in outputs:
            self.metrics[f'{prefix}/acc'].add(value=outputs['acc_main'].mean().item())
        if 'acc_mod' in outputs:
            self.metrics[f'{prefix}/acc_mod'].add(value=outputs['acc_mod'].mean().item())
        if 'cls_acc_main' in outputs and 'cls_acc_mod' in outputs and outputs['cls_acc_main'] is not None and outputs['cls_acc_mod'] is not None:
            self.metrics[f'{prefix}/cls_acc'].add(value=outputs['cls_acc_main'].mean().item())
            self.metrics[f'{prefix}/cls_acc_mod'].add(value=outputs['cls_acc_mod'].mean().item())
        for k, v in outputs.items():
            if k.startswith('l2_dist'):
                self.metrics[f'{prefix}/{k}'].add(value=v.mean().item())

    def run_epoch(self, loader, train, steps):
        if self.model_type == 'communication':
            budget_acc = torch.zeros(20)
            segment_acc = []
            total_cnt = 0
        if self.model_type == 'fgsbir' or (self.model_type == 'communication' and self.model.loss_type == 'fgsbir'):
            sketch_names = []
            sketch_features = []
            image_names = []
            images_features = []

        for idx, batch in enumerate(loader):
            for k in batch.keys():
                if k in ['x_raw', 'x_raw_aug', 'scale', 'scale_aug',
                         'translate', 'translate_aug', 'parameters_aug']:
                    if isinstance(batch[k], list):
                        batch[k] = batch[k][0]
                if k == 'n_strokes_aug':
                    batch[k] = torch.stack(batch[k], dim=1).view(-1)
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            if train:
                self.optimizer.zero_grad()
                ctx = nullcontext()
            else:
                self.model.eval()
                ctx = torch.no_grad()

            with ctx:
                outputs = self.run_model(batch, train)

            if train:
                outputs['loss'].mean().backward()
                self.optimizer.step()
            else:
                self.model.train()

            self.update_metrics(outputs, train=train)

            steps += 1

            if steps % 100 == 0 and train:
                log_metrics(self.metrics, self.tblogger, steps)
                self.metrics = defaultdict(Accumulator)

            if steps % self.log_every == 0:
                postfix = '' #if train else f'/{steps}'
                all_log_images = self.log_images(batch, outputs, train,
                                                 coords_type='x_raw' if 'pcoords_raw' in outputs else 'x')

                for phase, log_imgs, n_img in all_log_images:
                     img_tensor = F.interpolate(log_imgs, size=(self.view_scale, self.view_scale))
                     log_images(f'image_drawing_{phase}{postfix}', img_tensor, n_img, self.tblogger, steps if train else self.update_steps)

            if steps % self.save_every == 0:
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model_latest.pt'))

            if self.model_type == 'communication' and self.model.loss_type == 'class':
                if outputs['segments_per_sketch'] is None:
                    seq_lens = batch['n_strokes'].detach().cpu()
                    max_seq_lens = outputs['pcoords_n_strokes'].detach().cpu()
                else:
                    seq_lens = outputs['segments_per_sketch'].detach().cpu()
                    max_seq_lens = outputs['pcoords_n_strokes'].detach().cpu()

                acc_all = outputs['acc_main_all'].detach().cpu()
                budget = torch.arange(seq_lens.max()).unsqueeze(0).repeat(seq_lens.shape[0], 1)
                budget = (budget+1) / seq_lens.unsqueeze(1)
                for b in range(20):
                    curr_budget = ((b+1)/20)
                    if self.model.budget is not None and curr_budget > self.model.budget:
                        budget_acc[b] = budget_acc[b-1]
                        continue
                    budget_mask = budget >= curr_budget
                    budget_arg = budget_mask.float().argmax(dim=1)
                    budget_arg = torch.stack([budget_arg, max_seq_lens-1]).min(0)[0]
                    budget_acc[b] += acc_all.gather(1, budget_arg.unsqueeze(1)).sum()

                if not train:
                    for aa, bud, seq in zip (acc_all, budget, seq_lens):
                        if self.model.budget is None:
                            sl = seq
                        else:
                            sl = int(np.ceil(seq * self.model.budget))
                        segment_acc.append(list(zip(bud[:sl].numpy(), aa[:sl].numpy())))

                total_cnt += acc_all.shape[0]

            if self.model_type == 'fgsbir' or (self.model_type == 'communication' and self.model.loss_type == 'fgsbir'):
                sketch_features.extend(outputs['sketch_feature'].cpu())
                sketch_names.extend(batch['positive_name'])

                for name, img_feat in zip(batch['positive_name'], outputs['image_feature'].cpu()):
                    if name not in image_names:
                        image_names.append(name)
                        images_features.append(img_feat)

        if self.model_type == 'fgsbir' or (self.model_type == 'communication' and self.model.loss_type == 'fgsbir'):
            rank = []
            images_features = torch.stack(images_features)

            for name, sketch_feature in zip(sketch_names, sketch_features):
                position_query = image_names.index(name)

                if self.model.loss_type == 'cross_entropy':
                    distance = -torch.matmul(sketch_feature.unsqueeze(0), images_features.T)
                    target_distance = -torch.matmul(sketch_feature.unsqueeze(0), images_features[position_query].unsqueeze(1))
                else:
                    distance = F.pairwise_distance(sketch_feature.unsqueeze(0), images_features)
                    target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                          images_features[position_query].unsqueeze(0))

                rank.append(distance.le(target_distance).sum().item())

            rank = np.array(rank)
            top1 = (rank <= 1).sum() / rank.shape[0]
            top5 = (rank <= 5).sum() / rank.shape[0]
            top10 = (rank <= 10).sum() / rank.shape[0]
            prefix = 'train' if train else 'test'
            self.metrics[f'{prefix}/acc'].add(value=top1)
            self.metrics[f'{prefix}/acc5'].add(value=top5)
            self.metrics[f'{prefix}/acc10'].add(value=top10)

        if self.model_type == 'communication' and self.model.loss_type == 'class':
            prefix = 'train' if train else 'test'
            for b, acc in enumerate(budget_acc):
                self.metrics[f'{prefix}/budget_acc_{(b+1)/20}'].add(value=acc.item()/total_cnt)


        aggregated = log_metrics(self.metrics, self.tblogger, steps if train else self.update_steps)
        self.metrics = defaultdict(Accumulator)
        if not train:
            is_best = False
            if 'test/acc' in aggregated:
                current_acc = aggregated['test/acc']
                if self.best_result is None or self.best_result < current_acc:
                    self.best_result = current_acc
                    is_best = True

            else:
                current_loss = aggregated['test/loss']
                if self.best_result is None or self.best_result > current_loss:
                    self.best_result = current_loss
                    is_best = True

            if is_best:
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model_best_acc.pt'))
                if 'test/acc' in aggregated:
                    with open(os.path.join(self.log_dir, 'test_acc.txt'), 'w') as f:
                        f.write(f'Acc: {self.best_result}\n')
                        if 'test/acc5' in aggregated:
                            f.write(f'Acc 5: {aggregated["test/acc5"]}\n')
                        if 'test/acc10' in aggregated:
                            f.write(f'Acc 10: {aggregated["test/acc10"]}\n')
                        if self.model_type == 'communication' and self.model.loss_type == 'class':
                            f.write('Budget Acc: ' + ' '.join([str(ba.item()/total_cnt) for ba in budget_acc]))

                if self.model_type == 'communication' and self.model.loss_type == 'class':
                    segments = sorted(set([s for segs in segment_acc for s, _ in segs]))
                    seg_acc = np.zeros(len(segments))
                    for segs in segment_acc:
                        last_idx = 0
                        last_acc = 0.
                        for bud, acc in segs:
                            curr_idx = segments.index(bud)
                            seg_acc[last_idx:curr_idx] += last_acc
                            last_idx = curr_idx
                            last_acc = acc
                        seg_acc[last_idx:] += last_acc
                    #segment_acc = [[k, v] for k, v in sorted(segment_acc.items())]
                    #segment_acc = np.array(segment_acc)
                    seg_acc = seg_acc / total_cnt
                    with open(os.path.join(self.log_dir, 'budget_acc.txt'), 'w') as f:
                        for bud, acc in zip(segments, seg_acc):
                            f.write(f'{bud} {acc}\n')

        return steps

    def init_blit_manager(self):
        fig, ax = plt.subplots()
        self.blit_manager = BlitManager(ax, fig.canvas)
        fig.canvas.draw()

    def log_images(self, inputs, outputs, train, coords_type='x'):
        if self.blit_manager is None:
            self.init_blit_manager()

        if coords_type == 'x':
            x_input = inputs['x'].detach().cpu().numpy()
            if 'x_aug' in inputs:
                x_aug_input = inputs['x_aug'].detach().cpu().numpy()
            pcoords = outputs['pcoords'].detach().cpu().numpy()
        elif coords_type == 'x_raw':
            def preprocess_raw(inputs, base, prefix='', postfix=''):
                if isinstance(inputs[prefix+base+postfix], list):
                    x_raw = [x.detach().cpu().numpy() for x in inputs[prefix+base+postfix]]
                else:
                    x_raw = inputs[prefix+base+postfix].detach().cpu().numpy()
                scale = inputs[prefix+'scale'+postfix].detach().cpu().numpy()
                translate = inputs[prefix+'translate'+postfix].detach().cpu().numpy()
                seq_mask = inputs[prefix+'seq_mask'+postfix]
                if seq_mask is not None:
                    seq_mask= seq_mask.detach().cpu().numpy()
                if prefix+'prim_id'+postfix in inputs and self.model_type == 'pmn':
                    prim_id = inputs[prefix+'prim_id'+postfix].detach().cpu().numpy()
                    color_per_id = [PRIM_COLORS[PRIM_ORDER.index(prim_name)] for prim_name in self.model.primitive_names[:-1] + ['circle']]
                else:
                    prim_id = None
                offset = 0
                x_input = []
                colors = []
                for nstr in inputs[prefix+'n_strokes'+postfix]:
                    x_input.append((x_raw[offset:offset+nstr],
                                    scale[offset:offset+nstr],
                                    translate[offset:offset+nstr],
                                    seq_mask[offset:offset+nstr] if seq_mask is not None else seq_mask))
                    if prim_id is None:
                        colors.append(None)
                    else:
                        colors.append([color_per_id[pi.item()] for pi in prim_id[offset:offset+nstr]])
                    offset += nstr
                return x_input, colors
            x_input, _ = preprocess_raw(inputs, 'x_raw')
            if 'x_aug' in inputs or 'x_raw_aug' in inputs:
                x_aug_input, _ = preprocess_raw(inputs, 'x_raw', postfix='_aug')
            pcoords, pcoords_colors = preprocess_raw(outputs, 'raw', prefix='pcoords_')
        else:
            raise NotImplementedError

        gt = {aug: [] for aug in range(outputs['n_augment'])}
        pred = []
        in_img = []
        if 'x_img' in inputs:
            orig_img = []
        for idx in range(min(self.n_log_img, len(x_input))):
            if 'x_img' in inputs:
                orig_img.append(inputs['x_img'][idx].to('cpu'))
            in_img.append(torch.tensor(make_image_fast(self.blit_manager, x_input[idx], self.label_names[inputs['label'][idx]], 1.)))
            for aug in range(outputs['n_augment']):
                if 'cls_pred_mod' in outputs and outputs['cls_pred_mod'] is not None:
                    cls_name = self.label_names[outputs['cls_pred_mod'][outputs['n_augment']*idx+aug]]
                    cls_prob = outputs['cls_pred_mod_prob'][outputs['n_augment']*idx+aug]
                else:
                    cls_name = None
                    cls_prob = None
                gt[aug].append(torch.tensor(make_image_fast(self.blit_manager, x_aug_input[outputs['n_augment']*idx+aug], cls_name, cls_prob)))

            if self.model_type in ['pmn', 'classifier', 'fgsbir']:
                if 'cls_pred_main' in outputs and outputs['cls_pred_main'] is not None:
                    cls_name = self.label_names[outputs['cls_pred_main'][idx]]
                    cls_prob = outputs['cls_pred_main_prob'][idx]
                else:
                    cls_name = None
                    cls_prob = None
                pred.append(torch.tensor(make_image_fast(self.blit_manager, pcoords[idx], cls_name, cls_prob, color=pcoords_colors[idx])))
            elif self.model_type == 'communication':
                max_seq_len = (outputs['pcoords_n_strokes'][:min(self.n_log_img, len(x_input))]).max().item()
                pred_idx = []
                for seq_idx in range(max_seq_len):
                    if 'cls_pred_main' in outputs and outputs['cls_pred_main'] is not None:
                        cls_name = self.label_names[outputs['cls_pred_main'][idx][seq_idx]]
                        cls_prob = outputs['cls_pred_main_prob'][idx][seq_idx]
                    else:
                        cls_name = None
                        cls_prob = None
                    pred_idx.append(torch.tensor(make_image_fast(self.blit_manager, pcoords[idx], cls_name, cls_prob, max_seq_len=seq_idx+1)))
                pred.append(pred_idx)
            else:
                raise NotImplementedError

        if self.model_type == 'communication':
            pred = list(map(list, zip(*pred)))
            pred = [item for sublist in pred for item in sublist]

        log_imgs_train = [torch.stack(in_img,dim=0), torch.stack(pred,dim=0)] + [torch.stack(gt[aug],dim=0) for aug in range(outputs['n_augment'])]

        if self.model_type == 'fgsbir':
            inv_transform = transforms.Compose([
                transforms.Normalize(mean = [ 0., 0., 0. ],
                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                     std = [ 1., 1., 1. ]),
                transforms.Resize(480),
                transforms.Pad((80, 0)),
            ])
            log_imgs_train.append((inv_transform(inputs['positive_img']) * 255).byte().cpu())
            log_imgs_train.append((inv_transform(inputs['negative_img']) * 255).byte().cpu())

        log_imgs_train = torch.cat(log_imgs_train,dim=0)

        imgs =  [('train' if train else 'test', log_imgs_train, min(self.n_log_img, len(x_input)))]
        return imgs


    def run_model(self, inputs, train):
        outputs = self.model(**inputs)
        #outputs['n_augment'] = 1 if self.model_type == 'pmn' and not self.model.finetune_prims and (not self.model.train_prims or self.model.learned_prims) else 0
        outputs['n_augment'] = 0
        

        return outputs
