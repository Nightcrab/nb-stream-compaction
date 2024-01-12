# nb-stream-compaction

Implementation of parallel stream compaction using Numba CUDA.

| N    | gpu time (ms) | cpu time (ms) |
|------|---------------|---------------|
| 10^4 |   1.20        |     0.362     |
| 10^5 |   1.99        |     3.40      |
| 10^6 |   3.49        |    37.0       |
| 10^7 |  13.3         |   356.0       |
| 10^8 |  70.8         |  3533         |
| 10^9 | 673           | 35934         |

^ tested on machine with Ryzen 5 3600 and RTX 3090. 

Based on the following paper:

> Hughes, D. et al. (2013). InK‐Compact: In‐Kernel Stream Compaction and Its Application to Multi‐Kernel Data Visualization on General‐Purpose GPUs.