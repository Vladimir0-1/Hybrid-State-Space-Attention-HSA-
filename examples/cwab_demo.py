"""
CWAB Honest Benchmark
"""

import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from cwab import CWAB


class StandardAttention(nn.Module):
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


def benchmark_speed_memory(model, seq_len, device='cuda', num_iters=15):
    model = model.to(device)
    model.train()
    x = torch.randint(0, 1000, (1, seq_len)).to(device)

    for _ in range(3):
        out = model(x)
        loss = out.mean()
        loss.backward()
        model.zero_grad()

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

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


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    print("=" * 70)
    print("CWAB Honest Benchmark")
    print("=" * 70)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    configs = {
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 4,
        'vocab_size': 10000
    }

    results = {'standard': [], 'cwab': []}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}\n")

    for seq_len in tqdm(seq_lengths, desc="Benchmarking"):
        model_std = TinyTransformer(**configs, attention_class=StandardAttention, dropout=0.1)
        std_res = benchmark_speed_memory(model_std, seq_len, device)
        results['standard'].append(std_res)
        clear_cache()

        model_cwab = TinyTransformer(**configs, attention_class=CWAB, window_size=512, num_global_tokens=64)
        cwab_res = benchmark_speed_memory(model_cwab, seq_len, device)
        results['cwab'].append(cwab_res)
        clear_cache()

    print("\n Raw Benchmark Data:")
    print("=" * 80)
    print(f"{'Seq Len':>8} | {'Standard (ms)':>14} | {'CWAB (ms)':>11} | {'Speedup':>7} | {'Std Mem (MB)':>12} | {'CWAB Mem (MB)':>12}")
    print("-" * 80)
    for i, seq in enumerate(seq_lengths):
        std_mem = results['standard'][i]['peak_memory_mb']
        cwab_mem = results['cwab'][i]['peak_memory_mb']
        std_mem_str = f"{std_mem:.1f}" if std_mem else "N/A"
        cwab_mem_str = f"{cwab_mem:.1f}" if cwab_mem else "N/A"
        speedup = results['standard'][i]['mean_time_ms'] / results['cwab'][i]['mean_time_ms']
        print(f"{seq:8d} | {results['standard'][i]['mean_time_ms']:10.2f} ±{results['standard'][i]['std_time_ms']:.1f} | {results['cwab'][i]['mean_time_ms']:8.2f} ±{results['cwab'][i]['std_time_ms']:.1f} | {speedup:6.1f}x | {std_mem_str:>12} | {cwab_mem_str:>12}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.errorbar(seq_lengths, [r['mean_time_ms'] for r in results['standard']],
                 yerr=[r['std_time_ms'] for r in results['standard']], label='Standard', capsize=3)
    plt.errorbar(seq_lengths, [r['mean_time_ms'] for r in results['cwab']],
                 yerr=[r['std_time_ms'] for r in results['cwab']], label='CWAB', capsize=3)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Forward+Backward Time')
    plt.legend()
    plt.grid(True)

    if device == 'cuda':
        plt.subplot(1, 2, 2)
        plt.plot(seq_lengths, [r['peak_memory_mb'] for r in results['standard']], 'o-', label='Standard')
        plt.plot(seq_lengths, [r['peak_memory_mb'] for r in results['cwab']], 's-', label='CWAB')
        plt.xlabel('Sequence Length')
        plt.ylabel('Memory (MB)')
        plt.title('Peak Memory Usage')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('honest_benchmark.png', dpi=150)
    plt.show()

    print("\n Benchmark complete. Results saved to honest_benchmark.png")


if __name__ == "__main__":
    main()
