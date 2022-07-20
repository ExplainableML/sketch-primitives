import os
import datetime
from contextlib import nullcontext

import torch
import matplotlib.pyplot as plt
import numpy as np

from models import get_model
from models.communication import calc_segments_per_sketch
from utils.data import get_data
from utils.flags import parser
from utils.coords_utils import PRIM_ORDER
from tqdm import tqdm
from itertools import islice


def split_by_lengths(seq, num):
    it = iter(seq)
    for x in num:
        out = list(islice(it, x))
        if out:
            yield out
        else:
            return  # StopIteration
    remain = list(it)
    if remain:
        yield remain


if __name__ == '__main__':
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Add timestamp to logdir
    if args.log_name is None:
        LOG_DIR = os.path.join(args.log_dir,
                               datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        LOG_DIR = os.path.join(args.log_dir,
                               args.log_name)


    dataloader_coords, n_classes, label_names = get_data(
            args.data_dir, args.dataset, batch_size=args.batch_size,
            train=True,
            preprocessed_root=args.preprocessed_root,
            load_fgsbir_photos=args.load_fgsbir_photos,
            resample_strokes=not args.no_stroke_resampling,
            max_stroke_length=args.max_stroke_length,
            shuffle=False
    )

    test_dataloader_coords, _, _ = get_data(
            args.data_dir, args.dataset, batch_size=args.n_log_img,
            train=False,
            preprocessed_root=args.preprocessed_root,
            load_fgsbir_photos=args.load_fgsbir_photos,
            resample_strokes=not args.no_stroke_resampling,
            max_stroke_length=args.max_stroke_length,
            shuffle=False
            
    )

    if args.model_type == 'none':
        model = None
    else:
        model = get_model(args, 9, DEVICE)

    if args.test is not None and (not args.model_type == 'communication' or args.loss_model_type != 'none'):
        model.load_state_dict(torch.load(args.test))
        model.eval()

    """
    if args.model_type == 'pmn':
        if args.prim_subset is not None:
            model.init_primitives(PRIM_ORDER[:args.prim_subset] + ['dot'])
        else:
            model.init_primitives(model.primitive_names)
        model.to(DEVICE)
    """

    model_type = args.model_type
    if args.add_comm_model:
        args.model_type = 'communication'
        comm_model = get_model(args, 9, DEVICE)
        n_iters = 2
    else:
        comm_model = None
        n_iters = 1

    loaders = [('test', test_dataloader_coords), ('train', dataloader_coords)]
    # Preprocess strokes
    with torch.no_grad():
        for dset, dset_loader in loaders:
            new_coords_list = []
            for batch in tqdm(dset_loader, desc=f'Pre-processing data {dset}'):
                for k in batch.keys():
                    if k in ['x_raw', 'x_raw_aug', 'scale', 'scale_aug',
                             'translate', 'translate_aug', 'parameters_aug']:
                        if isinstance(batch[k], list):
                            batch[k] = batch[k][0]
                    if k == 'n_strokes_aug':
                        batch[k] = torch.stack(batch[k], dim=1).view(-1)
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to('cuda')

                orig_strokes = batch['n_strokes']
                for citer in range(n_iters):
                    strokes = batch['n_strokes']
                    if model is None:
                        points = list(batch['x_raw'].to('cpu').split(list(strokes), dim=0))
                        scale = list(batch['scale'].to('cpu').split(list(strokes), dim=0))
                        translate = list(batch['translate'].to('cpu').split(list(strokes), dim=0))
                        stroke_lens = batch['stroke_lens'].split(list(strokes), dim=0)
                        _, _, segments_per_sketch = calc_segments_per_sketch(batch['segments'], batch['stroke_lens'], batch['n_strokes'])
                        segments = batch['segments'].split(list(segments_per_sketch), dim=0)

                        new_coords_list += [{'x_raw': x.detach().cpu(),
                                             'label': l.item(),
                                             'n_strokes': n.detach().cpu(),
                                             'scale': s.detach().cpu(),
                                             'translate': t.detach().cpu(),
                                             'segments': se.detach().cpu(),
                                             'stroke_lens': sl.detach().cpu()}
                                            for x, l, n, s, t, se, sl in zip(points, batch['label'], strokes, scale, translate, segments, stroke_lens)]

                    else:
                        if comm_model is not None and citer == 1:
                            new_coords = comm_model(**batch)
                        else:
                            new_coords = model.draw(**batch)

                        if citer == 1 or model_type in ['pmn', 'communication']:
                            if comm_model is not None and citer == 0:
                                for k, v in new_coords.items():
                                    if k == 'pcoords_raw':
                                        batch['x_raw'] = v
                                    elif k.startswith('pcoords'):
                                        batch[k.replace('pcoords_', '')] = v
                                batch['n_strokes_orig'] = orig_strokes
                            else:
                                strokes = new_coords['pcoords_n_strokes']
                                if isinstance(new_coords['pcoords_raw'], list):
                                    offset = 0
                                    points = []
                                    for n_s in strokes:
                                        points.append([p.detach().cpu() for p in new_coords['pcoords_raw']][offset:offset+n_s])
                                        offset += n_s
                                else:
                                    points = list(new_coords['pcoords_raw'].detach().to('cpu').split(list(strokes), dim=0))
                                scale = list(new_coords['pcoords_scale'].to('cpu').split(list(strokes), dim=0))
                                translate = list(new_coords['pcoords_translate'].to('cpu').split(list(strokes), dim=0))

                                stroke_lens = batch['stroke_lens'].split(list(orig_strokes), dim=0)
                                _, _, segments_per_sketch = calc_segments_per_sketch(batch['segments'], batch['stroke_lens'], batch['n_strokes'])
                                segments = batch['segments'].split(list(segments_per_sketch), dim=0)
                                new_coords_list += [{'x_raw': x,
                                                     'label': l.item(),
                                                     'n_strokes': n.detach().cpu(),
                                                     'scale': s.detach().cpu(),
                                                     'translate': t.detach().cpu(),
                                                     'segments': se.detach().cpu(),
                                                     'stroke_lens': sl.detach().cpu()}
                                                    for x, l, n, s, t, se, sl in zip(points, batch['label'], strokes, scale, translate, segments, stroke_lens)]

                                if 'pcoords_prim_id' in new_coords:
                                    prim_ids = list(new_coords['pcoords_prim_id'].to('cpu').split(list(strokes), dim=0))
                                    for i, pi in enumerate(prim_ids):
                                        new_coords_list[-len(prim_ids):][i]['prim_id'] = pi.detach().cpu()

                        else:
                            sketches = list(split_by_lengths((new_coords), strokes))
                            split_scale = list(batch['scale'].split(list(strokes), dim=0))
                            split_translate = list(batch['translate'].split(list(strokes), dim=0))
                            new_strokes=torch.Tensor([]).to('cpu')
                            scales = torch.Tensor([]).to('cuda')
                            translates = torch.Tensor([]).to('cuda')
                            strokes = list(strokes)
                            for i in range(len(sketches)):
                                strokes[i] = 0
                                for j in range(len(sketches[i])):
                                    scales = torch.cat([scales, split_scale[i][j].repeat(len(sketches[i][j]), 1, 1)], dim=0)
                                    translates = torch.cat([translates, split_translate[i][j].repeat(len(sketches[i][j]), 1, 1)], dim=0)
                                    new_strokes = torch.cat([new_strokes, sketches[i][j]],dim=0)
                                    strokes[i]+= len(sketches[i][j])

                            if comm_model is not None:
                                batch['x_raw'] = new_strokes.to(batch['x_raw'])
                                batch['n_strokes'] = torch.tensor(strokes).to(batch['n_strokes'])
                                batch['scale'] = scales.to(batch['scale'])
                                batch['translate'] = translates.to(batch['translate'])
                                batch['n_strokes_orig'] = orig_strokes
                            else:
                                scale = list(scales.split(strokes, dim=0))
                                translate = list(translates.split(strokes, dim=0))
                                points = list(new_strokes.split(strokes, dim=0))
                                strokes = torch.tensor(strokes)
                                stroke_lens = batch['stroke_lens'].split(list(batch['n_strokes']), dim=0)
                                _, _, segments_per_sketch = calc_segments_per_sketch(batch['segments'], batch['stroke_lens'], batch['n_strokes'])
                                segments = batch['segments'].split(list(segments_per_sketch), dim=0)

                                new_coords_list += [{'x_raw': x,
                                                     'label': l.item(),
                                                     'n_strokes': n,
                                                     'scale': s.detach().cpu(),
                                                     'translate': t.detach().cpu(),
                                                     'segments': se.detach().cpu(),
                                                     'stroke_lens': sl.detach().cpu(),
                                                     'n_strokes_orig': n_orig.detach().cpu()}
                                                    for x, l, n, s, t, se, sl, n_orig in zip(points, batch['label'], strokes, scale, translate, segments, stroke_lens, orig_strokes)]

            dset_loader.dataset.set_preprocessed(new_coords_list)
            dset_loader.dataset.save_preprocessed(os.path.join(LOG_DIR, 'data'), save_images=not args.only_save_coords, suffix='.npy')
