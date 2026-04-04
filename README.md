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

📊 Raw Benchmark Data:
======================================================================
 Seq Len | Standard (ms) |   HSA (ms) | Speedup | Std Mem (MB) | HSA Mem (MB)
----------------------------------------------------------------------
     128 |      11.50 ±1.9 |    27.95 ±6.2 |    0.4x |       151.0 |       195.7
     256 |       8.93 ±0.4 |    51.26 ±21.7 |    0.2x |       217.5 |       218.0
     512 |      40.95 ±15.1 |    58.42 ±26.3 |    0.7x |       310.4 |       273.5
    1024 |      29.44 ±4.3 |    46.55 ±16.8 |    0.6x |       623.1 |       407.9
    2048 |      85.56 ±1.2 |    47.81 ±2.7 |    1.8x |      1729.5 |       624.3
    4096 |     294.89 ±2.7 |   100.93 ±1.0 |    2.9x |      6045.7 |      1060.7

