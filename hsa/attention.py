"""
Hybrid State-Space Attention (HSA) - Linear-complexity attention mechanism
Author: Vladimir0-1
License: MIT

This is the core attention layer that can replace standard multi-head attention.
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
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        window_size: Size of sliding window for local attention
        num_global_tokens: Number of compressed global tokens
        dropout: Dropout probability
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

        # Learnable global memory tokens (for compressed attention)
        self.global_memory = nn.Parameter(
            torch.randn(1, num_heads, num_global_tokens, self.head_dim)
        )

        # Compression via 1D convolution (stride=4 for 4x compression)
        self.compressor = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=4)

        # Adaptive mixing gates
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask (not used in current impl, kept for API compat)
        
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Sliding window attention (local context)
        local_out = self._sliding_window_attention(hidden_states)

        # 2. Compressed global attention (global context)
        global_out = self._compressed_global_attention(hidden_states)

        # 3. Information broadcast (exponential diffusion)
        broadcast_out = self._information_broadcast(hidden_states)

        # 4. Adaptive mixing of local and global
        mix_weights = self.mix_gate(torch.cat([local_out, global_out], dim=-1))
        mixed = mix_weights * local_out + (1 - mix_weights) * global_out
        mixed = mixed + broadcast_out

        return self.out_proj(self.dropout(mixed))

    def _sliding_window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Local attention within a sliding window. O(n * window_size)."""
        batch, seq, dim = x.shape
        window = min(self.window_size, seq)
        stride = window // 2

        # Pad to ensure we can cover the whole sequence
        if seq < window:
            return self._full_attention(x)
        
        # Unfold into windows
        pad_len = (stride - (seq - window) % stride) % stride
        x_pad = F.pad(x, (0, 0, 0, pad_len))
        windows = x_pad.unfold(1, window, stride).transpose(2, 3)  # (B, num_windows, window, D)

        # Reshape for batched attention
        B, num_windows, win_len, D = windows.shape
        windows_flat = windows.reshape(B * num_windows, win_len, D)
        
        # Multi-head reshape
        windows_flat = windows_flat.reshape(B * num_windows, win_len, self.num_heads, self.head_dim)
        windows_flat = windows_flat.transpose(1, 2)  # (B*W, heads, win_len, head_dim)
        
        # Self-attention within each window
        attn_weights = torch.matmul(windows_flat, windows_flat.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, windows_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B * num_windows, win_len, D)
        
        # Fold windows back
        return self._fold_windows(attn_out.reshape(B, num_windows, win_len, D), seq, window, stride)

    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback to full attention when sequence is shorter than window."""
        B, N, D = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, x)
        out = out.transpose(1, 2).reshape(B, N, D)
        return out

    def _compressed_global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Global context via sequence compression and learnable memory tokens."""
        batch, seq, dim = x.shape
        
        # Compress sequence to fixed number of tokens
        if seq >= 4:
            compressed = self.compressor(x.transpose(1, 2)).transpose(1, 2)
            compressed = compressed[:, :self.num_global_tokens, :]
        else:
            compressed = x
        
        # Add learnable memory tokens
        global_tokens = torch.cat([
            compressed,
            self.global_memory.expand(batch, -1, -1, -1).flatten(1, 2)
        ], dim=1)  # (B, num_global_tokens*2, D)
        
        # Multi-head reshape for global tokens
        global_tokens = global_tokens.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x_mh = x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-attention: each token attends to global tokens
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

    @staticmethod
    def _fold_windows(windows, original_len, window_size, stride):
        """Merge overlapping windows with triangular weighting."""
        B, num_windows, win_len, D = windows.shape
        output = torch.zeros(B, original_len, D, device=windows.device)
        weights = torch.zeros(original_len, 1, device=windows.device)

        for i in range(num_windows):
            start = i * stride
            end = min(start + window_size, original_len)
            length = end - start
            tri_weight = torch.linspace(0.5, 1.5, length, device=windows.device)
            output[:, start:end] += windows[:, i, :length] * tri_weight
            weights[start:end] += tri_weight

        return output / (weights + 1e-6)
