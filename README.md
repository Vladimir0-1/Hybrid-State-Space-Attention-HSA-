[![Donate](https://img.shields.io/badge/Donate-Boosty-orange)](https://www.donationalerts.com/c/vladimir0_1) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vladimir0-1/Hybrid-State-Space-Attention-HSA-/blob/main/examples/hsa_demo.ipynb)


# Hybrid State-Space Attention (HSA)

**An architectural pattern for linear-complexity attention with collective memory and dream-based self-improvement.**

Not a model. A **plug-and-play attention mechanism** that replaces standard multi-head attention in any transformer.



## Core Concepts

| Component                  | Function                                                     |
|----------------------------|--------------------------------------------------------------|
| **Compressed Attention**   | Global context via learnable centroids (k-means on-the-fly)  |
| **Sliding Window**         | Local precision with O(n×window) complexity                  |
| **Ring Broadcast**         | Exponential information diffusion, O(n log n)                |
| **Memory Tokens**          | Long-term storage across sequences                           |
| **Adaptive Mixing**        | Learnable gates balance local/global contributions           |
| **Dream Cycles**           | Agents sleep → hallucinated ideal self → improved strategies |
| **Collective Unconscious** | Shared memory pool across multiple instances                 |



## Complexity

| Operation         | Standard Attention | HSA        |
|-------------------|--------------------|------------|
| Per token         | O(n)               | O(1)       |
| Full sequence     | O(n²)              | **O(n)**   |
| Long context (1M) | ~1e12 FLOPs        | ~1e7 FLOPs |

**HSA is linear.** Always.


## 📊 Speed Benchmark

![HSA vs Standard Attention](honest_benchmark.png.png)

## 📊 Benchmark Results (T4 GPU)

| Seq Len | Standard (ms) | HSA (ms) | Speedup | Standard Mem (MB) | HSA Mem (MB) |
|---------|--------------|----------|---------|-------------------|---------------|
| 128 | 11.5 | 27.9 | 0.4x | 151 | 196 |
| 256 | 8.9 | 51.3 | 0.2x | 217 | 218 |
| 512 | 40.9 | 58.4 | 0.7x | 310 | 273 |
| 1024 | 29.4 | 46.5 | 0.6x | 623 | 408 |
| 2048 | 85.6 | 47.8 | **1.8x** | 1729 | 624 |
| 4096 | 294.9 | 100.9 | **2.9x** | 6046 | 1061 |

**Key takeaway:** HSA trades off some speed on short contexts for **6x lower memory** and **3x faster inference** on long contexts (4K+ tokens).

