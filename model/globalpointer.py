# -*- coding: utf-8 -*-
import sys

import torch
import torch.nn as nn


class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True):
        """
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        """
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE  # 旋转式位置编码
        self.trail_mask = tril_mask

        # LSTM
        # self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1,
        #                     bidirectional=True, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # transformer
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)

        # self-attention
        # self.nums_head = 8
        # self.input_dim = self.inner_dim * 2
        # self.dim_k = self.input_dim
        # self.dim_v = self.input_dim
        # assert self.dim_k % self.nums_head == 0
        # assert self.dim_v % self.nums_head == 0
        # self.q = nn.Linear(self.input_dim, self.dim_k)
        # self.k = nn.Linear(self.input_dim, self.dim_k)
        # self.v = nn.Linear(self.input_dim, self.dim_v)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, context_outputs, attention_mask):
        self.device = attention_mask.device
        last_hidden_state = context_outputs[0]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # original
        # outputs = self.dense(last_hidden_state)

        # # transformer
        outputs = self.transformer_encoder(last_hidden_state)
        outputs = self.dense(outputs)

        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)

        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5
