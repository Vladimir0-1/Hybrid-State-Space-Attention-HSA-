"""
CWAB - Compressed Window Attention Broadcast
Author: Vladimir0-1
License: MIT

A plug-and-play attention mechanism with linear complexity.
Combines sliding window, compressed global attention, and information broadcast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CWAB(nn.Module):
    """
    CWAB: Linear-complexity attention mechanism.
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        window_size: Size of sliding window for local attention
        num_global_tokens: Number of compressed global tokens
        dropout: Dropout probability
        short_seq_threshold: Use full attention below this length (avoid overhead)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 512,
        num_global_tokens: int = 64,
        dropout: float = 0.1,
        short_seq_threshold: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = min(window_size, hidden_size)
        self.num_global_tokens = num_global_tokens
        self.short_seq_threshold = short_seq_threshold

        # Learnable global memory tokens
        self.global_memory = nn.Parameter(torch.randn(1, num_global_tokens, hidden_size))
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Compression (4x stride)
        self.compressor = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=4)
        
        # Adaptive mixing gate
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask (not used, kept for API compatibility)
        
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch, seq, dim = x.shape
        
        # For short sequences, use full attention to avoid overhead
        if seq <= self.short_seq_threshold:
            return self._full_attention(x)
        
        # For long sequences, use hybrid approach
        local_out = self._sliding_window(x)
        global_out = self._global_context(x)
        
        # Adaptive mixing
        mix = self.mix_gate(torch.cat([local_out, global_out], dim=-1))
        out = mix * local_out + (1 - mix) * global_out
        
        return self.out_proj(self.dropout(out))

    def _sliding_window(self, x: torch.Tensor) -> torch.Tensor:
        """Sliding window attention with non-overlapping windows."""
        batch, seq, dim = x.shape
        window = min(self.window_size, seq)
        
        if seq <= window:
            return self._full_attention(x)

        # Pad to make divisible
        pad = (window - seq % window) % window
        if pad > 0:
            x_pad = F.pad(x, (0, 0, 0, pad))
        else:
            x_pad = x
            
        padded_seq = x_pad.shape[1]
        n_windows = padded_seq // window

        # Reshape into windows
        windows = x_pad.reshape(batch, n_windows, window, dim)
        windows = windows.reshape(batch * n_windows, window, self.num_heads, self.head_dim)
        windows = windows.transpose(1, 2)

        # Self-attention within each window
        attn = torch.matmul(windows, windows.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, windows)

        # Reshape back
        out = out.transpose(1, 2).reshape(batch * n_windows, window, dim)
        out = out.reshape(batch, n_windows, window, dim)
        out = out.reshape(batch, padded_seq, dim)
        
        return out[:, :seq, :] if pad > 0 else out

    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full attention (fallback for short sequences)."""
        batch, seq, dim = x.shape
        
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        return out.transpose(1, 2).reshape(batch, seq, dim)

    def _global_context(self, x: torch.Tensor) -> torch.Tensor:
        """Compressed global attention via centroids and learnable memory."""
        batch, seq, dim = x.shape
        
        # Compress sequence
        if seq >= 4 and self.num_global_tokens > 0:
            compressed = self.compressor(x.transpose(1, 2)).transpose(1, 2)
            compressed = compressed[:, :self.num_global_tokens, :]
        else:
            compressed = x

        # Add learnable memory tokens
        memory = self.global_memory.expand(batch, -1, -1)
        global_tokens = torch.cat([compressed, memory], dim=1)
        
        # Multi-head cross-attention
        global_tokens = global_tokens.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x_mh = x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(x_mh, global_tokens.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, global_tokens)
        
        return context.transpose(1, 2).reshape(batch, seq, dim)
