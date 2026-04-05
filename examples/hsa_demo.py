"""
HSSA Honest Benchmark - Run this script to compare Standard vs HSSA
No cherry-picking. Multiple metrics. Draw your own conclusions.

Usage: python examples/hssa_demo.py
"""

import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from hssa import HybridStateSpaceAttention

class StandardAttention(nn.Module):
    """Standard multi-head attention (baseline)"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class TinyTransformer(nn.Module):
    """Minimal transformer for fair comparison"""
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, attention_class, **attn_kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_size),
                'attn': attention_class(hidden_size, num_heads, **attn_kwargs),
                'norm2': nn.LayerNorm(hidden_size),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
            }) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            x = layer['attn'](x)
            x = residual + x
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = residual + x
        x = self.norm(x)
        return self.lm_head(x)


def benchmark_speed_memory(model, seq_len, device='cuda', num_iters=20):
    """Measure forward + backward time and peak memory"""
    model = model.to(device)
    model.train()
    x = torch.randint(0, 1000, (1, seq_len)).to(device)
    
    # Warmup
    for _ in range(5):
        out = model(x)
        loss = out.mean()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
    torch.cuda.synchronize() if device == 'cuda' else None
    
    times = []
    for _ in range(num_iters):
        start = time.time()
        out = model(x)
        loss = out.mean()
        loss.backward()
        model.zero_grad()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else None
    
    return {
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'peak_memory_mb': peak_mem
    }


def main():
    print("="*70)
    print("HSSA Honest Benchmark")
    print("="*70)
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    configs = {
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 4,
        'vocab_size': 10000
    }
    
    results = {'standard': [], 'hssa': []}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}\n")
    
    for seq_len in tqdm(seq_lengths, desc="Benchmarking"):
        # Standard model
        model_std = TinyTransformer(
            **configs,
            attention_class=StandardAttention,
            dropout=0.1
        )
        std_res = benchmark_speed_memory(model_std, seq_len, device)
        
        # HSSA model
        model_hsa = TinyTransformer(
            **configs,
            attention_class=HybridStateSpaceAttention,
            window_size=512,
            num_global_tokens=64
        )
        hssa_res = benchmark_speed_memory(model_hssa, seq_len, device)
        
        results['standard'].append(std_res)
        results['hssa'].append(hssa_res)
    
    # Print raw data
    print("\n Raw Benchmark Data:")
    print("="*80)
    print(f"{'Seq Len':>8} | {'Standard (ms)':>14} | {'HSSA (ms)':>11} | {'Speedup':>7} | {'Std Mem (MB)':>12} | {'HSSA Mem (MB)':>12}")
    print("-"*80)
    for i, seq in enumerate(seq_lengths):
        std_mem_str = f"{results['standard'][i]['peak_memory_mb']:.1f}" if results['standard'][i]['peak_memory_mb'] else "N/A"
        hsa_mem_str = f"{results['hssa'][i]['peak_memory_mb']:.1f}" if results['hssa'][i]['peak_memory_mb'] else "N/A"
        speedup = results['standard'][i]['mean_time_ms'] / results['hssa'][i]['mean_time_ms']
        print(f"{seq:8d} | {results['standard'][i]['mean_time_ms']:10.2f} ±{results['standard'][i]['std_time_ms']:.1f} | {results['hssa'][i]['mean_time_ms']:8.2f} ±{results['hssa'][i]['std_time_ms']:.1f} | {speedup:6.1f}x | {std_mem_str:>12} | {hssa_mem_str:>12}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    std_times = [r['mean_time_ms'] for r in results['standard']]
    hsa_times = [r['mean_time_ms'] for r in results['hssa']]
    std_err = [r['std_time_ms'] for r in results['standard']]
    hsa_err = [r['std_time_ms'] for r in results['hssa']]
    
    axes[0].errorbar(seq_lengths, std_times, yerr=std_err, fmt='o-', label='Standard', capsize=5)
    axes[0].errorbar(seq_lengths, hssa_times, yerr=hssa_err, fmt='s-', label='HSSA', capsize=5)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Forward+Backward Time (lower is better)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if device == 'cuda':
        std_mem = [r['peak_memory_mb'] for r in results['standard']]
        hssa_mem = [r['peak_memory_mb'] for r in results['hssa']]
        axes[1].plot(seq_lengths, std_mem, 'o-', label='Standard')
        axes[1].plot(seq_lengths, hssa_mem, 's-', label='HSSA')
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Peak Memory (MB)')
        axes[1].set_title('GPU Memory Usage (lower is better)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Memory benchmark requires CUDA', ha='center')
        axes[1].set_title('Memory (unavailable)')
    
    plt.tight_layout()
    plt.savefig('honest_benchmark.png', dpi=150)
    plt.show()
    
    print("\n Benchmark complete. Results saved to honest_benchmark.png")
    print("\nInterpretation is left to the reader.")


if __name__ == "__main__":
    main()
