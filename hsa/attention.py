"""
Hybrid State-Space Attention (HSA) - Linear-complexity attention mechanism
Author: Vladimir0-1
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridStateSpaceAttention(nn.Module):
    """
    HSA: Linear-complexity attention with sliding window, compressed global context,
    information broadcast, and adaptive mixing.

    Complexity: O(n) in sequence length.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 512,
        num_global_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens

        # Learnable global memory tokens
        self.global_memory = nn.Parameter(
            torch.randn(1, num_heads, num_global_tokens, self.head_dim)
        )

        # Compression via 1D convolution
        self.compressor = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=4)

        # Adaptive mixing gates
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Sliding window attention
        local_out = self._sliding_window_attention(hidden_states)

        # 2. Compressed global attention
        global_out = self._compressed_global_attention(hidden_states)

        # 3. Information broadcast
        broadcast_out = self._information_broadcast(hidden_states)

        # 4. Adaptive mixing
        mix_weights = self.mix_gate(torch.cat([local_out, global_out], dim=-1))
        mixed = mix_weights * local_out + (1 - mix_weights) * global_out
        mixed = mixed + broadcast_out

        return self.out_proj(self.dropout(mixed))

    def _sliding_window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Local attention within a sliding window."""
        batch, seq, dim = x.shape
        window = min(self.window_size, seq)
        stride = window // 2

        if seq <= window:
            return self._full_attention(x)

        # Pad to make divisible
        pad_len = (stride - (seq - window) % stride) % stride
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        padded_seq = x_pad.shape[1]

        # Unfold into windows
        windows = x_pad.unfold(1, window, stride)  # (B, num_windows, window, D)
        B, num_windows, win_len, D = windows.shape

        # Reshape for multi-head attention
        windows = windows.reshape(B * num_windows, win_len, self.num_heads, self.head_dim)
        windows = windows.transpose(1, 2)  # (B*W, heads, win_len, head_dim)

        # Self-attention within each window
        attn_weights = torch.matmul(windows, windows.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, windows)
        attn_out = attn_out.transpose(1, 2).reshape(B * num_windows, win_len, D)

        # Fold windows back
        return self._fold_windows(attn_out, B, seq, window, stride, padded_seq)

    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback to full attention for short sequences."""
        B, N, D = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, x)
        out = out.transpose(1, 2).reshape(B, N, D)
        return out

    def _compressed_global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Global context via sequence compression."""
        batch, seq, dim = x.shape

        # Compress sequence
        if seq >= 4:
            compressed = self.compressor(x.transpose(1, 2)).transpose(1, 2)
            compressed = compressed[:, :self.num_global_tokens, :]
        else:
            compressed = x

        # Add learnable memory tokens
        global_tokens = torch.cat([
            compressed,
            self.global_memory.expand(batch, -1, -1, -1).flatten(1, 2)
        ], dim=1)

        # Multi-head reshape
        global_tokens = global_tokens.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x_mh = x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention
        attn_weights = torch.matmul(x_mh, global_tokens.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        global_context = torch.matmul(attn_weights, global_tokens)
        global_context = global_context.transpose(1, 2).reshape(batch, seq, dim)

        return global_context

    def _information_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential information diffusion: O(n log n)."""
        batch, seq, dim = x.shape
        result = torch.zeros_like(x)
        current = x
        stride = 1

        while stride < seq:
            left = torch.roll(current, shifts=stride, dims=1)
            right = torch.roll(current, shifts=-stride, dims=1)
            current = current + 0.5 * (left + right)
            result = result + current
            stride *= 2

        return result / (math.log2(seq) + 1) if seq > 1 else result

    def _fold_windows(self, attn_out, batch, original_len, window_size, stride, padded_seq):
        """Merge overlapping windows with triangular weighting."""
        # attn_out: (B * num_windows, win_len, D)
        num_windows = attn_out.shape[0] // batch
        win_len = attn_out.shape[1]
        D = attn_out.shape[2]

        attn_out = attn_out.reshape(batch, num_windows, win_len, D)

        output = torch.zeros(batch, original_len, D, device=attn_out.device)
        weights = torch.zeros(original_len, 1, device=attn_out.device)

        for i in range(num_windows):
            start = i * stride
            end = min(start + window_size, original_len)
            length = end - start
            if length <= 0:
                continue

            tri_weight = torch.linspace(0.5, 1.5, length, device=attn_out.device)
            output[:, start:end] += attn_out[:, i, :length] * tri_weight.unsqueeze(0).unsqueeze(-1)
            weights[start:end] += tri_weight.unsqueeze(-1)

        # Avoid division by zero
        weights = weights.clamp(min=1e-6)
        return output / weights
