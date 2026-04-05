"""
CWAB - Compressed Window Attention Broadcast
Author: Vladimir0-1
License: MIT

Linear-complexity attention mechanism:
- Compressed global context
- Windowed local attention
- Information broadcast
- Adaptive mixing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CWAB(nn.Module):
    def __init__(self, hidden_size, num_heads=8, window_size=512, num_global_tokens=64, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = min(window_size, hidden_size)
        self.num_global_tokens = num_global_tokens
        self.short_seq_threshold = 1024

        self.global_memory = nn.Parameter(torch.randn(1, num_global_tokens, hidden_size))
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.compressor = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=4)
        self.mix_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch, seq, dim = x.shape

        if seq <= self.short_seq_threshold:
            return self._full_attention(x)

        local_out = self._sliding_window(x)
        global_out = self._global_context(x)

        mix = self.mix_gate(torch.cat([local_out, global_out], dim=-1))
        out = mix * local_out + (1 - mix) * global_out

        return self.out_proj(self.dropout(out))

    def _sliding_window(self, x):
        batch, seq, dim = x.shape
        window = min(self.window_size, seq)

        if seq <= window:
            return self._full_attention(x)

        pad = (window - seq % window) % window
        x_pad = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        padded_seq = x_pad.shape[1]
        n_windows = padded_seq // window

        windows = x_pad.reshape(batch, n_windows, window, dim)
        windows = windows.reshape(batch * n_windows, window, self.num_heads, self.head_dim)
        windows = windows.transpose(1, 2)

        attn = torch.matmul(windows, windows.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, windows)

        out = out.transpose(1, 2).reshape(batch * n_windows, window, dim)
        out = out.reshape(batch, n_windows, window, dim)
        out = out.reshape(batch, padded_seq, dim)
        return out[:, :seq, :] if pad > 0 else out

    def _full_attention(self, x):
        batch, seq, dim = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(batch, seq, dim)

    def _global_context(self, x):
        batch, seq, dim = x.shape

        if seq >= 4 and self.num_global_tokens > 0:
            compressed = self.compressor(x.transpose(1, 2)).transpose(1, 2)
            compressed = compressed[:, :self.num_global_tokens, :]
        else:
            compressed = x

        memory = self.global_memory.expand(batch, -1, -1)
        global_tokens = torch.cat([compressed, memory], dim=1)

        global_tokens = global_tokens.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x_mh = x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(x_mh, global_tokens.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, global_tokens)

        return context.transpose(1, 2).reshape(batch, seq, dim)
