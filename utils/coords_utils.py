import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.axes import Axes

PRIM_COLORS = ['#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#0072B2', '#785EF0']
PRIM_ORDER = ['halfcircle', 'circle', 'line', 'corner', 'triangle', 'square', 'u']

def get_stroke_stats(strokes, scale_margin=0.01):
    mins = strokes.min(dim=-2, keepdim=True)[0]
    maxs = strokes.max(dim=-2, keepdim=True)[0]
    sizes = (maxs - mins)
    scale = sizes.max(dim=-1, keepdim=True)[0]
    scale += scale * scale_margin
    # replace 0. scales with margin
    scale = scale.masked_fill(scale.isclose(torch.tensor([0.], device=scale.device)), scale_margin)
    translation = (mins + maxs) / 2.
    aspect = sizes / scale
    return scale, translation, aspect

def normalize_strokes(strokes):
    scale, translation, _ = get_stroke_stats(strokes)
    strokes = (strokes - translation) / scale * 2.
    return strokes, scale, translation

def preprocess_transformer(seq, resample_strokes=False, resample_size=25,
                           points_per_primitive=3):
    assert seq[0, 2] == 1., seq[0]

    strokes = split_seq_into_strokes(seq, max_seq_len=None if resample_strokes else resample_size)
    segments_len = [int(np.ceil(len(s)/points_per_primitive)) for s in strokes]
    if resample_strokes:
        strokes = [resample_stroke(s, resample_size) for s in strokes]
        strokes = torch.stack(strokes)
        strokes = (strokes[:, :, :2] - 127.5) / 127.5
        strokes, scale, translation = normalize_strokes(strokes)
        stroke_lens = tuple([torch.tensor(resample_size).unsqueeze(0) for _ in strokes])
        mask = torch.zeros(strokes.shape[:2], dtype=torch.bool)
        position = torch.linspace(-1., 1., steps=resample_size).unsqueeze(0).repeat(len(strokes), 1)
    else:
        strokes = [(s[:, :2] - 127.5) / 127.5 for s in strokes]
        strokes, scale, translation = zip(*[normalize_strokes(s) for s in strokes])
        scale, translation = torch.stack(scale), torch.stack(translation)
        stroke_lens = tuple([torch.tensor(len(s)).unsqueeze(0) for s in strokes])
        strokes_tensor = torch.zeros((len(strokes), resample_size, 2))
        mask = torch.ones(strokes_tensor.shape[:2], dtype=torch.bool)
        position = torch.zeros(strokes_tensor.shape[:2])
        for i, s in enumerate(strokes):
            slen = len(s)
            strokes_tensor[i, :slen] = s
            mask[i, :slen] = False
            position[i, :slen] = torch.linspace(-1., 1., steps=slen)
        strokes = strokes_tensor

    segments = []
    for i, sl in enumerate(segments_len):
        if resample_strokes:
            sl = min(resample_size, sl)
        seg = []
        cnt = stroke_lens[i].item()
        for n_remain in range(sl, 0, -1):
            seg.append(int(np.round(cnt / n_remain)))
            cnt -= seg[-1]
        segments.append(seg)

    return strokes, len(strokes), scale, translation, segments, stroke_lens, mask, position

def preprocess_sequence(seq, Nmax=99):
    assert seq[0, 2] == 1., seq[0]
    nonzero = torch.nonzero(seq.sum(-1) == 0.)
    if nonzero.nelement() == 0:
        len_seq = Nmax
    else:
        len_seq = nonzero[0].item() - 1
    if seq.shape[0] < Nmax+1:
        seq = torch.cat([seq, torch.zeros(Nmax+1-seq.shape[0], 4)])
    new_seq = (seq[:,:2]- 127.5) / 127.5
    new_seq = new_seq[1:] - new_seq[:-1]
    new_seq = torch.cat([new_seq, seq[1:, 2:], torch.zeros((Nmax, 1))], dim=1)
    new_seq[len_seq-1:,4] = 1
    new_seq[len_seq-1:,2:4] = 0
    new_seq[len_seq:,0:2] = 0
    return new_seq, len_seq


def split_seq_into_strokes(orig_seq, max_seq_len=None):
    split_idx = [0]+list(torch.where(orig_seq[:,-1]>0)[0]+1)
    split_idx = [(split_idx[i+1]-split_idx[i]).item() for i in range(len(split_idx)-1)]
    pad_len = orig_seq.shape[0] - sum(split_idx)
    if pad_len > 0:
        split_idx.append(pad_len)

    current_strokes = list(torch.split(orig_seq, split_idx))
    if max_seq_len is not None:
        updated_strokes = []
        for stroke in current_strokes:
            if stroke.shape[0] > max_seq_len and not (stroke[0] == 0.).all().item():
                n_splits = int(np.ceil(stroke.shape[0] / max_seq_len))
                if stroke.shape[0] > (max_seq_len*n_splits-n_splits+1):
                    n_splits += 1

                new_strokes = []
                offset = 0
                for n_remain in range(n_splits, 0, -1):
                    stroke_len = int(np.round((stroke.shape[0]+n_remain-1-offset) / n_remain))
                    new_strokes.append(stroke[offset:offset+stroke_len])
                    offset = offset + stroke_len - 1

                updated_strokes.extend(new_strokes)
            else:
                updated_strokes.append(stroke)
        current_strokes = updated_strokes

    is_padded = (current_strokes[-1][0] == 0.).all().item()
    if is_padded:
        current_strokes = current_strokes[:-1]

    pnt_strokes = []
    for i, stroke in enumerate(current_strokes):
        if stroke.shape[0] == 1:
            pnt_strokes.append(i)
    for i in reversed(pnt_strokes):
        del current_strokes[i]

    return current_strokes


def resample_strokes(current_strokes, resample_ratios, max_points):
    avail_points = max_points - sum([c.shape[0] for c in current_strokes])
    new_strokes = []
    for curr_stroke in current_strokes:
        n_pnts = curr_stroke.shape[0]
        ratio = np.random.rand() * (resample_ratios[1] - resample_ratios[0]) + resample_ratios[0]
        while (new_n_pnts := max(2, int(np.around(n_pnts * ratio)))) - n_pnts > avail_points:
            ratio = np.random.rand() * (resample_ratios[1] - resample_ratios[0]) + resample_ratios[0]
        new_strokes.append(resample_stroke(curr_stroke.clone(), new_n_pnts))
        avail_points -= new_n_pnts - n_pnts
    return new_strokes


def make_image_fast(blit_manager, sequence, class_name=None, class_prob=None, no_axis=False, color=None, max_seq_len=None, linewidth=3, enlarge=1.0, arbitrary_mask=False):
    if isinstance(sequence, tuple):
        # raw stroke-based coordinates
        seq, scale, translate, seq_mask = sequence
        if isinstance(seq, list):
            strokes = [(se / 2 * sc + tr) * enlarge for se, sc, tr in zip(seq, scale, translate)]
        else:
            strokes = (seq / 2 * scale + translate) * enlarge
        check_end = False
    else:
        # relative coordinates
        sequence[:, 0] = np.cumsum(sequence[:, 0])
        sequence[:, 1] = np.cumsum(sequence[:, 1])
        """plot drawing with separated strokes"""
        strokes = np.split(sequence, np.where(sequence[:, 3] + sequence[:, 4] > 0)[0] + 1)
        strokes[0] = np.concatenate([strokes[0][0][np.newaxis], strokes[0]], axis=0)
        strokes[0][0, :2] = 0.
        check_end = True
        seq_mask = None

    if max_seq_len is not None:
        if max_seq_len > len(strokes):
            strokes = []
        else:
            strokes = strokes[:max_seq_len]

    if isinstance(blit_manager, Axes):
        ax = blit_manager
        artists = []
    else:
        ax = blit_manager.ax
        artists = blit_manager.get_artists()
    if len(artists) == 0:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        if no_axis:
            ax.set_axis_off()
        txt = ax.text(-2.2, 2.2, '', size=20, animated=True)
        blit_manager.add_artist(txt)
    else:
        txt = artists[0]

    if not no_axis and class_name is not None and class_prob is not None:
        txt.set_text(f'{class_name} ({class_prob:.2%})')
    else:
        txt.set_text('')

    if arbitrary_mask and seq_mask is not None:
        new_strokes = []
        for s, m in zip(strokes, seq_mask):
            used = np.argwhere(~m).flatten()
            stroke_sections = []
            start = None
            for u in used:
                if start is None:
                    start = u
                elif u != last_u + 1:
                    stroke_sections.append((start, last_u))
                    start = u
                last_u = u
            if start is not None:
                stroke_sections.append((start, last_u))

            for sec_start, sec_end in stroke_sections:
                new_strokes.append(s[sec_start:sec_end+1])

        strokes = new_strokes
        seq_mask = None

    n_strokes = 0
    for i, s in enumerate(strokes):
        if s.size == 0:
            break

        x = s[:, 0]
        y = -s[:, 1]
        if seq_mask is not None:
            seq_len = np.argwhere(seq_mask[i]).flatten()
            if len(seq_len) > 0:
                x = x[:seq_len[0]]
                y = y[:seq_len[0]]

        if (x == x[0]).all() and (y == y[0]).all():
            marker = 'o'
        else:
            marker = ''

        if color is not None:
            if isinstance(color, list):
                c = color[i]
            else:
                c = color

        if i+1 < len(artists):
            artists[i+1].set_data(x, y)
            artists[i+1].set_marker(marker)
            if color is not None:
                artists[i+1].set_color(c)
        else:
            if color is not None:
                (line,) = ax.plot(x, y, marker=marker, linewidth=linewidth, animated=True, color=c)
            else:
                (line,) = ax.plot(x, y, marker=marker, linewidth=linewidth, animated=True)
            blit_manager.add_artist(line)

        n_strokes += 1
        if check_end and s[-1, 4] == 1:
            break

    # don't show unused artists
    for i in range(n_strokes+1, len(artists)):
        artists[i].set_data([], [])

    if not isinstance(blit_manager, Axes):
        blit_manager.update()
        # grab the pixel buffer and dump it into a numpy array
        img = np.array(blit_manager.canvas.renderer.buffer_rgba())[:, :, :3]
        img = np.moveaxis(img, -1, 0)

        return img


def resample_data(edge_list, n=20, random=True, endpoint=False):
    total_edge_length = sum([e[0] for e in edge_list])
    if random:
        locations = sorted(np.random.rand(n) * total_edge_length)
    else:
        if endpoint:
            locations = np.linspace(0., total_edge_length, num=n)
        else:
            locations = np.linspace(0., total_edge_length, num=n+1)[:-1]

    points = []
    loc_i = 0
    curr_length = 0
    for edge in edge_list:
        start_pnt = edge[1]
        end_pnt = edge[2]
        edge_length = edge[0]
        while loc_i < n and (rel_length := locations[loc_i] - curr_length) <= edge_length:
            if edge_length == 0.:
                pnt = start_pnt
            else:
                pnt = start_pnt + rel_length/edge_length * (end_pnt - start_pnt)
            points.append(pnt)
            loc_i += 1

        curr_length += edge[0]

    points = np.array(points)
    assert (~np.isnan(points)).all()

    return points

def resample_stroke(stroke, n):
    pnts = stroke[:, :2].numpy()
    edge_list = [[np.linalg.norm(pnts[i] - pnts[i+1]), pnts[i], pnts[i+1]] for i in range(pnts.shape[0]-1)]
    re_pnts = resample_data(edge_list, n, random=False, endpoint=True)
    new_stroke = torch.from_numpy(re_pnts)
    states = torch.zeros(new_stroke.shape, dtype=new_stroke.dtype)
    states[:, 0] = 1.
    states[-1, 0] = 0.
    states[-1, 1] = 1.
    new_stroke = torch.cat([new_stroke, states], dim=1)
    return new_stroke
