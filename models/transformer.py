import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StrokeEncoderTransformer(nn.Module):
    def __init__(self, Nz=128, enc_hidden_size=128, dropout=0.1, stroke_len=25,
                 embed_sketch=True, n_layer_strokes=3, n_layer_sketch=3,
                 use_pos_embed=True, use_sinusoid_embed=False, fourier_scale=1.,
                 out_dim=None, use_embed_token=True, return_points_embed=False):
        super().__init__()
        if out_dim is None:
            out_dim = Nz
        self.enc_hidden_size = enc_hidden_size
        self.embed_sketch = embed_sketch
        self.stroke_len = stroke_len
        self.use_pos_embed = use_pos_embed
        self.sinusoid_embed = use_sinusoid_embed
        self.use_embed_token = use_embed_token
        self.return_points_embed = return_points_embed
        # Transformer:
        encoder_layer = nn.TransformerEncoderLayer(
            enc_hidden_size, 8, dim_feedforward=4*enc_hidden_size,
            dropout=dropout, activation='gelu'
        )
        self.stroke_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layer_strokes
        )
        self.fc_in = nn.Linear(2, enc_hidden_size)
        self.fc_out = nn.Linear(enc_hidden_size, out_dim)
        if self.use_pos_embed and not self.sinusoid_embed:
            self.pos_embed = nn.Parameter(torch.randn(stroke_len+1, 1, enc_hidden_size))
        else:
            self.embed_token = nn.Parameter(torch.randn(1, 1, enc_hidden_size))
            if self.use_pos_embed:
                self.fc_pos_embed = nn.Linear(enc_hidden_size//(2*3)*2, enc_hidden_size)

        if self.sinusoid_embed:
            randn = fourier_scale * torch.randn((1, enc_hidden_size//(2*3)))
            #randn = fourier_scale * torch.randn((3, enc_hidden_size//2))
            self.register_buffer('fourier_randn', randn)

        if self.embed_sketch:
            if self.sinusoid_embed:
                self.fc_transform = nn.Linear((enc_hidden_size//(2*3))*2*3, enc_hidden_size)
                #self.fc_transform = nn.Linear(enc_hidden_size, enc_hidden_size)
            else:
                self.fc_transform = nn.Linear(3, enc_hidden_size)
            self.sketch_transformer = nn.TransformerEncoder(encoder_layer,
                                                            num_layers=n_layer_sketch)
            self.sketch_embed = nn.Parameter(torch.randn(1, enc_hidden_size))
        # active dropout:
        self.train()

    def pad_stroke_sequence(self, inputs, stroke_lens):
        max_stroke_len = max(stroke_lens)
        offset = 0
        strokes = []
        mask = torch.ones((len(stroke_lens), max_stroke_len+1), device=inputs.device, dtype=torch.bool)
        mask[:, -1] = False
        if self.use_pos_embed:
            positions = torch.zeros(max_stroke_len, (len(stroke_lens)), device=inputs.device)
        else:
            positions = None
        for i, sl in enumerate(stroke_lens):
            strokes.append(torch.cat([inputs[offset:offset+sl],
                                      torch.zeros((max_stroke_len-sl, 2), device=inputs.device)]))
            mask[i, :sl] = False
            if self.use_pos_embed:
                positions[:sl, i] = torch.linspace(-1, 1., steps=sl, device=inputs.device)
            offset += sl
        return torch.stack(strokes, dim=1), mask, max_stroke_len, positions

    def forward(self, inputs, n_strokes=None, scale=None, translate=None, stroke_lens=None, seq_mask=None, position=None, mask=None, points_mask=None, **kwargs):
        if inputs.dim() == 2:
            inputs, seq_mask, slen, pos = self.pad_stroke_sequence(inputs.transpose(0, 1), stroke_lens)
            if points_mask is None:
                points_mask = seq_mask
            else:
                points_mask = points_mask | seq_mask
        elif seq_mask is not None:
            slen = inputs.shape[0]
            pos = position.transpose(0, 1)
            seq_mask = torch.cat([seq_mask, torch.zeros_like(seq_mask[:, :1])], dim=1)
            if points_mask is None:
                points_mask = seq_mask
            else:
                points_mask = points_mask | seq_mask
        else:
            slen = self.stroke_len
        embed = self.fc_in(inputs)
        if self.use_pos_embed:
            if self.sinusoid_embed:
                pos_embed = (2.*np.pi*pos.unsqueeze(-1)) @ self.fourier_randn
                pos_embed = torch.cat([torch.sin(pos_embed), torch.cos(pos_embed)], dim=-1)
                pos_embed = self.fc_pos_embed(pos_embed)
                pos_embed = torch.cat([pos_embed, self.embed_token.expand(1, pos_embed.shape[1], pos_embed.shape[2])])
            else:
                pos_embed = self.pos_embed
        else:
            pos_embed = torch.cat([torch.zeros((slen, 1, self.enc_hidden_size), device=embed.device), self.embed_token])
        embed = torch.cat([embed, torch.zeros((1, embed.size(1), embed.size(2)), device=embed.device)]) + pos_embed
        embed = self.stroke_transformer(embed, src_key_padding_mask=points_mask)
        if self.return_points_embed:
            points_embed = embed[:-1]
        embed = embed[-1]

        if self.embed_sketch:
            transform = torch.cat([scale, translate], dim=-1).squeeze(1)
            if self.sinusoid_embed:
                transform = (2.*np.pi*transform.unsqueeze(-1)) @ self.fourier_randn
                #transform = (2.*np.pi*transform) @ self.fourier_randn
                transform = torch.cat([torch.sin(transform), torch.cos(transform)], dim=-1)
                transform = transform.flatten(-2)
            transform_embed = self.fc_transform(transform)
            embed = embed + transform_embed

            embed_batch = torch.split(embed, n_strokes.tolist())
            seq_len = n_strokes.max()
            batch = []
            padding_mask = []
            for eb in embed_batch:
                pad_len = seq_len-eb.size(0)
                padding = torch.zeros((pad_len, eb.size(1)), device=eb.device)
                embd = torch.cat([eb, padding])
                if self.use_embed_token:
                    embd = torch.cat([embd, self.sketch_embed])
                batch.append(embd)
                if mask is None:
                    pad_mask = torch.zeros(seq_len+(1 if self.use_embed_token else 0),
                                           device=eb.device, dtype=torch.bool)
                    if self.use_embed_token:
                        pad_mask[-pad_len-1:-1] = True
                    elif pad_len != 0:
                        pad_mask[-pad_len:] = True
                    padding_mask.append(pad_mask)

            batch = torch.stack(batch, dim=1)
            if mask is None:
                padding_mask = torch.stack(padding_mask, dim=0)
            else:
                padding_mask = mask
            embed = self.sketch_transformer(batch, src_key_padding_mask=padding_mask)
            if self.use_embed_token:
                embed = embed[-1]
        else:
            padding_mask = None

        if self.return_points_embed:
            return self.fc_out(embed), points_embed, padding_mask
        else:
            return self.fc_out(embed), padding_mask
