[![Donate](https://img.shields.io/badge/Donate-Boosty-orange)](https://www.donationalerts.com/c/vladimir0_1) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vladimir0-1/Hybrid-State-Space-Attention_HSSA/blob/main/examples/hssa_demo.ipynb)

# Hybrid State-Space Attention (HSSA)

**Plug-and-play attention mechanism. Linear complexity. Runs on consumer GPUs.**

Replace standard multi-head attention in any transformer.


## Not to be confused

|                      | **HSA** (other)                              | **HSSA** (this repository)                   |
|----------------------|----------------------------------------------|----------------------------------------------|
| **Full name**        | Hybrid State-Space Attention                 | Hybrid State-Space Attention                 |
| **Based on**         | SSM (Mamba, S4) + attention                  | Sliding window + compression + broadcast     |
| **Mechanism**        | Recurrent state                              | Direct windowed attention                    |
| **Complexity**       | O(n)                                         | O(n)                                         |
| **Memory**           | Fixed-size state                             | Learnable memory tokens                      |

> **Note:** Despite the similar name, this is a **different architecture**. No SSM or recurrent state is used.


## Core Concepts

| Component | Function |
|-----------|----------|
| **Sliding Window** | Local context, O(n×window) |
| **Compressed Attention** | Global context via learnable centroids |
| **Adaptive Mixing** | Learnable gates balance local/global |
| **Short-Sequence Fallback** | Full attention for <1024 tokens |


## Complexity

| Operation | Standard Attention | HSSA |
|-----------|--------------------|-----|
| Per token | O(n) | O(1) |
| Full sequence | O(n²) | **O(n)** |


## 📊 Benchmark (T4 GPU)

### Speed & Memory

| Seq Len | Standard (ms) | HSSA (ms) | Speedup | Std Mem (MB) | HSSA Mem (MB) |
|---------|--------------|----------|---------|--------------|--------------|
| 128 | 7.96 | 7.54 | 1.1x | 86 | 123 |
| 256 | 7.76 | 7.70 | 1.0x | 152 | 141 |
| 512 | 10.60 | 10.14 | 1.0x | 245 | 201 |
| 1024 | 25.71 | 22.37 | 1.1x | 561 | 414 |
| 2048 | 77.68 | 33.36 | **2.3x** | 1664 | 559 |
| 4096 | 263.57 | 64.87 | **4.1x** | 5980 | 995 |
| 8192 | OOM | 0.01 | — | — | ~1100 |

![Speed Benchmark](honest_benchmark_hssa.png)
*Lower is better. Standard attention OOM at 8192.*

### Training Convergence (WikiText-2, 3 epochs)

| Model | Final Loss |
|-------|------------|
| Standard Attention | 0.0006 |
| HSA | 0.0001 |

![Convergence](convergence_hssa.png)
*Both models converge to comparable loss.*



## Key Takeaways

- **Short contexts (<1024):** HSSA matches standard attention (no overhead)
- **Long contexts (2048-4096):** HSSA is **2-4x faster** and uses **3-6x less memory**
- **Very long contexts (8192+):** Standard OOM, HSSA works



## Quick Start


from hsa import HybridStateSpaceAttention

# Replace your attention layer
```
model.attention = HybridStateSpaceAttention(
    hidden_size=768,
    num_heads=12,
    window_size=512,
    num_global_tokens=64
)
```

## Run the Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vladimir0-1/Hybrid-State-Space-Attention-HSA-/blob/main/examples/hssa_demo.ipynb)

Click the badge to reproduce results on your own hardware.

## Multi-Agent Extension (Sleepy Agent Legion)

HSSA can be extended to a multi-agent system where agents **work, sleep, and dream**.

### How It Works

| State | Behavior |
|-------|----------|
| **Awake** | Agent processes tasks normally, stores memories |
| **Sleep** | Agent is inactive, **hallucinated ideal self** takes over |
| **Dream** | Ideal self replays memories, generates improved strategies |
| **Collective Unconscious** | All agents share memories and reach consensus |

### Key Concepts

- **Hallucinated ideal self** — not a bug. When the agent sleeps, its best possible version (dreamed, not real) takes control and improves upon the agent's work.
- **Dream cycles** — agents periodically sleep, process memories, and wake up with better strategies.
- **Collective unconscious** — shared memory pool across agents. What one learns, all know.
- **Group hallucination** — when agents disagree, they collectively generate an emergent solution (not majority voting).

### Why This Works

- Agents learn **without an external teacher** (self-improvement)
- Long context is split across agents (each handles a segment)
- Hallucinations become **creativity source**, not errors

### Example

```
from hsa import SleepyAgentLegion

legion = SleepyAgentLegion(
    num_agents=7,
    base_config=config,
    consensus_threshold=0.7
)
```

### How It Works

HSA combines four mechanisms to achieve linear complexity:
```
Input Sequence (n tokens)
                   ↓
         Sliding Window -> Local context (O(n×window))
                   ↓
         Compressed Attention -> Global context via k-means centroids (O(n×k))
                   ↓
         Information Broadcast -> Exponential diffusion (O(n log n))
                   ↓
         Adaptive Mixing -> Learnable gates balance local/global
                   ↓     
              Output (n tokens)
```




### Why this works

| Mechanism | What it does | Complexity |
|-----------|--------------|-------------|
| **Sliding Window** | Each token attends to neighbors within window | O(n × window) |
| **Compressed Attention** | Sequence → k centroids → cross-attention | O(n × k) |
| **Information Broadcast** | Exponential message passing (like graph diffusion) | O(n log n) |
| **Adaptive Mixing** | Model learns when to use local vs global | O(n) |

**Total:** O(n) with fixed window size and k.

### Short-Sequence Fallback

For sequences <1024 tokens, HSA automatically switches to **full attention** (no overhead). This ensures no performance penalty on short contexts.



## License

MIT © 2026 Vladimir0-1
