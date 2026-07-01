import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
import time

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, kdim, qdim, vdim, heads=8, dim_head=64, dropout=0., print_layer=0):
        super().__init__()
        dim = qdim
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(kdim, inner_dim, bias = False)
        self.to_v = nn.Linear(vdim, inner_dim, bias = False)
        self.to_q = nn.Linear(qdim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.print_layer = print_layer

    def forward(self, q, k=None, v=None): # , print_anchor = None, print_radar = None, print_depth = None
        if k == None or v == None:
            k,v = q,q
        b, n, _, h = *q.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        kqv = self.to_k(k), self.to_q(q), self.to_v(v)
        k, q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kqv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(kdim=dim, qdim=dim, vdim=dim, heads=heads, dim_head=dim_head, dropout=0., print_layer=i))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
    def forward(self, x):
        # x, anchors, radar, depth = x
        for attn, ff in self.layers:
            x = attn(x) #, print_depth = depth, print_radar = radar, print_anchor = anchors
            x = ff(x)
        return x

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        可学习的位置编码
        Args:
            d_model: 嵌入维度
            max_len: 最大序列长度
        """
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]