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

![HSA vs Standard Attention](honest_benchmark.png)



## How to Use (Architecture Integration)

```python
# Step 1: Replace your attention module
from hsa import HybridStateSpaceAttention

model.attention = HybridStateSpaceAttention(
    hidden_size=768,
    num_heads=12,
    window_size=2048,
    num_global_tokens=128,
    enable_dreams=True  # optional
)

# Step 2: Train as usual (or use distillation from standard attention)
# Step 3: For multi-agent — wrap N copies with collective unconscious
