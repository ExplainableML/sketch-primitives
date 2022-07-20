import os
import collections
from itertools import chain, combinations
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import copy

from .coords_utils import preprocess_sequence, preprocess_transformer, make_image_fast
from .transformations import transform_primitives, ToTransformMatrix
from .plt_blit import BlitManager

from torch.utils.data._utils.collate import default_collate


def sketch_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: sketch_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence) and not isinstance(elem, str):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem = next(it)
        elem_size = len(elem)
        elem_dims = elem[0].dim() if isinstance(elem[0], torch.Tensor) else 0
        if not all(len(elem) == elem_size for elem in it) or elem_dims > 1 or isinstance(elem, tuple):
            batch = sum(batch, tuple())

        if elem_dims > 0:
            return torch.cat(batch)
        else:
            transposed = zip(*batch)
            return [sketch_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)




class PreprocessedSketchDataset(data.Dataset):
    def __init__(self, coordinate_path_root, sketch_list, idx_class=0,
                 original_strokes_list=None, load_fgsbir_photos=False,
                 original_path_root=None):
        self.coordinate_path_root = coordinate_path_root
        self.load_fgsbir_photos = load_fgsbir_photos

        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()

            self.coordinate_urls, sketch_url_list = self.get_coords_list(coordinate_path_root, sketch_url_list)

            self.labels = [int(sketch_url.strip().split(' ')[1])
                           for sketch_url in sketch_url_list]
            label_names = [sketch_url.strip().split(' ')[0].split('/')[idx_class + 1]
                           for sketch_url in sketch_url_list]

            if self.load_fgsbir_photos:
                assert original_path_root is not None
                alt_coordinate_urls, _ = self.get_coords_list(os.path.join(original_path_root, 'picture_files'), sketch_url_list)
                self.picture_urls = [re.sub('_[^_]*\.pt', '.png', cu.replace('/train/', '/').replace('/test/', '/')) for cu in alt_coordinate_urls]
                self.unique_picture_urls = set(self.picture_urls)

                self.img_transform = transforms.Compose([
                    transforms.Resize(299), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

            self.original_strokes = None

            self.label_names = np.unique(label_names)

        self.n_classes = len(self.label_names)
        self.keys = ['x_raw', 'scale', 'translate', 'seq_mask', 'position']



    def __len__(self):
        return len(self.coordinate_urls)

    def get_coords_list(self, coordinate_path_root, sketch_url_list):
        return [os.path.join(coordinate_path_root, (sketch_url.strip(
        ).split(' ')[0]).replace('.npy', '.pt')) for sketch_url in sketch_url_list], sketch_url_list


    def __getitem__(self, item):
        coordinate = torch.load(self.coordinate_urls[item])
        assert coordinate['label'] == self.labels[item]

        if isinstance(coordinate['x_raw'], list):
            split_scale = list(coordinate['scale'].split(1, dim=0))
            split_translate = list(coordinate['translate'].split(1, dim=0))
            coordinate['n_strokes'] = 0
            for i in range(len(coordinate['x_raw'])):
                split_scale[i] = split_scale[i].repeat(coordinate['x_raw'][i].shape[0],1,1)
                split_translate[i] = split_translate[i].repeat(coordinate['x_raw'][i].shape[0],1,1)
                coordinate['n_strokes']+=coordinate['x_raw'][i].shape[0]
            coordinate['x_raw'] = torch.cat(coordinate['x_raw'],dim=0).float()
            coordinate['scale'] = torch.cat(split_scale,dim=0).float()
            coordinate['translate'] = torch.cat(split_translate,dim=0).float()

        for k in coordinate.keys():
            if isinstance(coordinate[k], np.ndarray):
                coordinate[k] = torch.from_numpy(coordinate[k]).float()
            elif k == 'scale' and isinstance(coordinate[k], float):
                coordinate[k] = torch.tensor([[[coordinate[k]]]])
            elif k == 'stroke_lens' and isinstance(coordinate[k], int):
                coordinate[k] = torch.tensor([coordinate[k]])

        coordinate['seq_mask'] = torch.zeros(coordinate['x_raw'].shape[:2], dtype=torch.bool)
        coordinate['position'] = torch.linspace(-1., 1., steps=coordinate['x_raw'].shape[1]).unsqueeze(0).repeat(coordinate['x_raw'].shape[0], 1)
        data_dict = {k: coordinate[k].split(1, dim=0) for k in self.keys}
        data_dict['label'] = coordinate['label']
        data_dict['n_strokes'] = coordinate['n_strokes']
        if isinstance(coordinate['segments'], int):
            data_dict['segments'] = tuple([torch.tensor(coordinate['segments']).unsqueeze(0)])
        else:
            data_dict['segments'] = tuple([c.unsqueeze(0) for c in coordinate['segments']])
        data_dict['stroke_lens'] = tuple([s.unsqueeze(0) for s in coordinate['stroke_lens']])

        if 'prim_id' in coordinate:
            if isinstance(coordinate['prim_id'], int):
                coordinate['prim_id'] = torch.tensor([[[coordinate['prim_id']]]])
            data_dict['prim_id'] = coordinate['prim_id'].split(1, dim=0)

        if 'n_strokes_orig' in coordinate:
            data_dict['n_strokes_orig'] = coordinate['n_strokes_orig']

        if self.load_fgsbir_photos:
            # TODO: duplicate code
            positive_path = self.picture_urls[item]
            uniq_urls = set(self.unique_picture_urls)
            uniq_urls.remove(positive_path)
            negative_path = np.random.choice(list(uniq_urls))
            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')
            positive_img = self.img_transform(positive_img)
            negative_img = self.img_transform(negative_img)
            data_dict['positive_img'] = positive_img
            data_dict['negative_img'] = negative_img
            data_dict['positive_name'] = positive_path.split('/')[-1].split('.')[0]

        if self.original_strokes is not None:
            data_dict['original_n_strokes'] = self.original_strokes[item]
        data_dict['name'] = item
        return data_dict

    def save_img(self, path, suffix='.pt', process=True):
        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        ax.set_axis_off()
        fig.add_axes(ax)
        self.blit_manager = BlitManager(ax, fig.canvas)
        fig.canvas.draw()
        for item in tqdm(range(len(self.coordinate_urls)), desc='Saving images'):
            file_path = self.coordinate_urls[item]
            relative_path = os.path.relpath(file_path, self.coordinate_path_root)
            file_path = os.path.join(path, relative_path).replace(suffix, '.png')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            coordinate = torch.load(self.coordinate_urls[item])
            sequence_raw = coordinate['x_raw']
            scale = coordinate['scale']
            translate = coordinate['translate']

            if isinstance(sequence_raw, list):
                sequence_img = torch.cat(
                    [((sequence_raw[i] / sequence_raw[i].abs().max()) / 2 * scale[i] + translate[i]) for i in
                     range(len(sequence_raw))], dim=0).numpy()
            else:
                sequence_img = (sequence_raw / 2 * scale + translate).numpy()
            x_img = make_image_fast(self.blit_manager, (sequence_img, 3.0, 0.0, None), no_axis=True, color='black')
            im = Image.fromarray(np.moveaxis(x_img, 0, -1).astype(np.uint8))
            im.save(file_path)

    def add_strokes_to_list(self, file_name):
        full_text = ""
        for item in tqdm(range(len(self.coordinate_urls)), desc='Updating file'):
            data = self.__getitem__(item)
            n_strokes = data['n_strokes']
            label = data['label']
            file_path = self.coordinate_urls[item]
            relative_path = os.path.relpath(file_path, self.coordinate_path_root)
            text = relative_path + ' ' + str(label) + ' ' + str(n_strokes.item())
            full_text+=text+'\n'

        with open(file_name, 'w') as file:
            file.write(full_text)

    def save_preprocessed(self, root, suffix='.pt'):
        self.save_img(os.path.join(root, 'imgs'), suffix)
        #self.save_coords(os.path.join(root, 'coords'), suffix)


class SketchDataset(data.Dataset):
    def __init__(self, coordinate_path_root, sketch_list,
                 idx_class=0, preprocessed=None,
                 Nmax=99, load_fgsbir_photos=False,
                 points_per_primitives=3, resample_strokes=False,
                 max_stroke_length=25):
        self.coordinate_path_root = coordinate_path_root
        self.preprocessed = preprocessed
        self.Nmax = Nmax
        self.load_fgsbir_photos = load_fgsbir_photos
        self.points_per_primitive = points_per_primitives
        self.resample_strokes = resample_strokes
        self.max_stroke_length = max_stroke_length

        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()

        self.coordinate_urls, sketch_url_list = self.get_coords_list(coordinate_path_root, sketch_url_list)

        self.labels = [int(sketch_url.strip().split(' ')[-1])
                       for sketch_url in sketch_url_list]
        label_names = [sketch_url.strip().split(' ')[0].split('/')[idx_class + 1]
                       for sketch_url in sketch_url_list]
        self.label_names = np.unique(label_names)

        if self.load_fgsbir_photos:
            self.sketch_picture_urls = [re.sub('.npy', '.png', cu.replace('coordinate_files', 'sketch_picture_files')) for cu in self.coordinate_urls]
            self.picture_urls = [re.sub('_[^_]*\.npy', '.png', cu.replace('coordinate_files', 'picture_files').replace('/train/', '/').replace('/test/', '/')) for cu in self.coordinate_urls]
            self.unique_picture_urls = set(self.picture_urls)

            self.img_transform = transforms.Compose([
                transforms.Resize(299), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                #transforms.RandomHorizontalFlip(),
            ])

        self.n_classes = len(self.label_names)

    def __len__(self):
        return len(self.coordinate_urls)

    def set_preprocessed(self, preprocessed):
        print('Setting preprocessed')
        self.preprocessed = preprocessed

    def get_coords_list(self, coordinate_path_root, sketch_url_list):
        return [os.path.join(coordinate_path_root, (sketch_url.strip(
        ).split(' ')[0]).replace('png', 'npy')) for sketch_url in sketch_url_list], sketch_url_list

    def read_coordinates(self, coordinate_urls, idx, normalize=False):
        coordinate = np.load(coordinate_urls[idx], encoding='latin1', allow_pickle=True)

        if coordinate.dtype == 'object':
            coordinate = coordinate[0]

        coordinate = coordinate.astype('float32')
        coordinate = torch.from_numpy(coordinate)
        if normalize:
            coordinate[:, :2] = ((coordinate[:, :2] - coordinate[:, :2].min()) / (
                        coordinate[:, :2] - coordinate[:, :2].min()).max()) * 255

        return coordinate

    def save_img(self, path, suffix='.svg', process=True):
        fig, ax = plt.subplots(figsize=(2.99, 2.99), constrained_layout=True)
        ax.set_axis_off()
        fig.add_axes(ax)
        self.blit_manager = BlitManager(ax, fig.canvas)
        fig.canvas.draw()
        for item in tqdm(range(len(self.coordinate_urls)), desc='Saving images'):
            file_path = self.coordinate_urls[item]
            relative_path = os.path.relpath(file_path, self.coordinate_path_root)
            file_path = os.path.join(path, relative_path).replace(suffix, '.png')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            coordinate = self.read_coordinates(self.coordinate_urls, item)
            sequence_raw = self.preprocessed[item]['x_raw']
            scale = self.preprocessed[item]['scale']
            translate = self.preprocessed[item]['translate']

            x_img = make_image_fast(self.blit_manager, (sequence_raw, scale, translate, None), no_axis=True, color='black', linewidth=1, enlarge=2.0)
            im = Image.fromarray(np.moveaxis(x_img, 0, -1).astype(np.uint8))
            im.save(file_path)

    def save_coords(self, path, suffix='.svg'):
        for item in tqdm(range(len(self.coordinate_urls)), desc='Saving coords'):
            file_path = self.coordinate_urls[item]
            relative_path = os.path.relpath(file_path, self.coordinate_path_root)
            file_path = os.path.join(path, relative_path).replace(suffix, '.pt')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            coordinate = self.preprocessed[item]
            for k in coordinate.keys():
                if isinstance(coordinate[k], torch.Tensor):
                    if coordinate[k].numel() == 1:
                        coordinate[k] = coordinate[k].item()
                    else:
                        coordinate[k] = coordinate[k].detach().cpu().numpy()
            coordinate['path'] = relative_path
            torch.save(coordinate, file_path)

    def add_strokes_to_list(self, file_name):
        full_text = ""
        for item in tqdm(range(len(self.coordinate_urls)), desc='Updating file'):
            coordinate = self.read_coordinates(self.coordinate_urls, item)
            file_path = self.coordinate_urls[item]
            relative_path = os.path.relpath(file_path, self.coordinate_path_root)
            _, n_strokes, _, _ = preprocess_transformer(coordinate)
            text = relative_path + ' ' + str(self.labels[item]) + ' ' + str(n_strokes)
            full_text+=text+'\n'
        with open(file_name, 'w') as file:
            file.write(full_text)


    def save_preprocessed(self, root, suffix='.svg', save_images=True):
        if save_images:
            self.save_img(os.path.join(root, 'imgs'), suffix)
        self.save_coords(os.path.join(root, 'coords'), suffix)

    def __getitem__(self, item):
        label = self.labels[item]

        coordinate = self.read_coordinates(self.coordinate_urls, item)

        sequence, length = preprocess_sequence(coordinate, Nmax=self.Nmax)
        sequence_raw, n_strokes, scale, translate, segments, stroke_lens, mask, position = preprocess_transformer(
            coordinate, resample_strokes=self.resample_strokes,
            points_per_primitive=self.points_per_primitive,
            resample_size=self.max_stroke_length
        )
        if self.preprocessed is not None:
            sequence_raw = self.preprocessed[item]
            # TODO: check if all calculated values match with preprocessed
            assert False

        sequence_raw = sequence_raw.split(1, dim=0)
        scale = scale.split(1, dim=0)
        translate = translate.split(1, dim=0)
        position = position.split(1, dim=0)
        if mask is not None:
            mask = mask.split(1, dim=0)

        data_dict = {
            'x': sequence,
            'length': length,
            'label': label,
            'x_raw': sequence_raw,
            'n_strokes': n_strokes,
            'scale': scale,
            'translate': translate,
            'segments': tuple([torch.tensor(s).unsqueeze(0) for seg in segments for s in seg]),
            'stroke_lens': stroke_lens,
            'seq_mask': mask,
            'position': position,
            'data_id': item
        }

        if self.load_fgsbir_photos:
            positive_path = self.picture_urls[item]
            sketch_img_path = self.sketch_picture_urls[item]
            uniq_urls = set(self.unique_picture_urls)
            uniq_urls.remove(positive_path)
            negative_path = np.random.choice(list(uniq_urls))
            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')
            if os.path.exists(sketch_img_path):
                sketch_img = Image.open(sketch_img_path).convert('RGB')
            else:
                sketch_img = None
            positive_img = self.img_transform(positive_img)
            negative_img = self.img_transform(negative_img)
            if sketch_img is not None:
                sketch_img = self.img_transform(sketch_img)
                data_dict['sketch_img'] = sketch_img
            else:
                data_dict['sketch_img'] = -1
            data_dict['positive_img'] = positive_img
            data_dict['negative_img'] = negative_img
            data_dict['positive_name'] = positive_path.split('/')[-1].split('.')[0]

        return data_dict


def get_data(data_root='./data', name='MNIST', batch_size=128, train=True,
             shuffle=True,
             preprocessed_root=None,
             original_strokes_list=None, load_fgsbir_photos=False,
             resample_strokes=False, max_stroke_length=25):

    if name == 'quickdraw':
        data_dir = os.path.join(data_root, 'quickdraw')

        data_dir_coords = os.path.join(data_dir, 'coordinate_files')
        data_dir = os.path.join(data_dir, 'picture_files')
        if train:
            sketch_list = os.path.join(data_dir_coords, 'train.txt')
            if preprocessed_root is not None:
                data_coords = PreprocessedSketchDataset(coordinate_path_root=preprocessed_root,
                                                       sketch_list=sketch_list,
                                                       original_strokes_list=original_strokes_list)
            else:
                data_coords = SketchDataset(
                    data_dir_coords, sketch_list=sketch_list,
                    resample_strokes=resample_strokes,
                    max_stroke_length=max_stroke_length
                )
        else:
            sketch_list = os.path.join(data_dir_coords, 'test.txt')
            if preprocessed_root is not None:
                data_coords = PreprocessedSketchDataset(coordinate_path_root=preprocessed_root,
                                                       sketch_list=sketch_list,
                                                       original_strokes_list=original_strokes_list)
            else:
                data_coords = SketchDataset(
                    data_dir_coords, sketch_list=sketch_list,
                    resample_strokes=resample_strokes,
                    max_stroke_length=max_stroke_length
                )

    elif name == 'quickdraw09':
        data_dir_coords = os.path.join(data_root, 'quickdraw09')

        if train:
            sketch_list = os.path.join(data_dir_coords, 'train.txt')

            if preprocessed_root is not None:
                data_coords = PreprocessedSketchDataset(
                    coordinate_path_root=preprocessed_root,
                    sketch_list=sketch_list,
                    original_strokes_list=original_strokes_list)
            else:
                data_coords = SketchDataset(
                    data_dir_coords, sketch_list=sketch_list,
                    Nmax=160,
                    resample_strokes=resample_strokes,
                    max_stroke_length=max_stroke_length
                )
        else:
            sketch_list = os.path.join(data_dir_coords, 'test.txt')
            if preprocessed_root is not None:
                data_coords = PreprocessedSketchDataset(
                    coordinate_path_root=preprocessed_root,
                    sketch_list=sketch_list,
                    original_strokes_list=original_strokes_list)
            else:
                data_coords = SketchDataset(
                    data_dir_coords, sketch_list=sketch_list,
                    Nmax=160,
                    resample_strokes=resample_strokes,
                    max_stroke_length=max_stroke_length
                )


    elif name in ['chairv2', 'shoev2']:
        data_dir = os.path.join(data_root, name)
        data_dir_coords = os.path.join(data_dir, 'coordinate_files')

        if train:
            sketch_list = os.path.join(data_dir_coords, 'train.txt')
        else:
            sketch_list = os.path.join(data_dir_coords, 'test.txt')

        if preprocessed_root is not None:
            data_coords = PreprocessedSketchDataset(
                coordinate_path_root=preprocessed_root, sketch_list=sketch_list,
                original_strokes_list=original_strokes_list,
                load_fgsbir_photos=load_fgsbir_photos, original_path_root=data_dir)
        else:
            data_coords = SketchDataset(
                data_dir_coords, sketch_list=sketch_list,
                Nmax=520, load_fgsbir_photos=load_fgsbir_photos,
                resample_strokes=resample_strokes,
                max_stroke_length=max_stroke_length
            )

    else:
        raise NotImplementedError

    if not train:
        generator = torch.Generator()
        generator.manual_seed(42)
    else:
        generator = None

    # Initialize dataloaders
    loader_coords = torch.utils.data.DataLoader(data_coords, batch_size, shuffle=shuffle, num_workers=8,
                                                generator=generator, collate_fn=sketch_collate)

    return loader_coords, data_coords.n_classes, data_coords.label_names
