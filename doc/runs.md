# Some statistics from fineweb runs

All train runs use fineweb dataset, setup is similar to the one used in llm.c Batch size 1M. All runs use same learning rate 0.01. All runs are fp8, MFU is computed relative to H100 theoretical 1950 TFlops, 6 flops per parameter for backprop.

## 1.5B active parameters, different sparsity
Model dim string e1792h14d42ffn20xp512moes6moe192
| | 1.5B (dense) | 3.3B | 5.6B | 10.4B | 20B|
|--|--|--|--|--|--|
|Experts|-|6 of 24|6 of 48|6 of 96|6 of 192|
|PP (GPU/instance)|1|1|2|4|8|
|MFU (distributed)|18.1|15|13.6|10.7|7.7|
|MFU (1 host)|20.5|16.5|15.7|12.15|8.8|
|Hellaswag|49.47|52.86|53.28|54.82|55.39|
|Logloss|2.465|2.407|2.382|2.345|2.317|
Performance is far from optimized in this code. Performance is mostly affected by high PP. PP can be radically reduced by sharding master weights across instances (not implemented). 
## MFU depending on sparsity
Model dim string e1536h12d16ffn18xp384moes6moe192, 500m active. Fits on single GPU.
| | 500M (dense)|1B|1.5B|2.7B|5.1B|
|--|--|--|--|--|--|
|Experts|-|6 of 24|6 of 48|6 of 96|6 of 192|
|MFU|20.55|16.22|15.68|14.5|12.55|

## MFU depending on PP
Model e2048xp512h16d36ffn24moe96moes6, 6 of 96 experts. It fits on 2 GPU, but we spread it on 4 and 8 GPUs as well
|1.6B of 10B|PP=2|PP=4|PP=8|
|--|--|--|--|
|MFU|14.33|12.94|10.23|

## Longer runs
1.5B dense model
|1.5B (dense)|Hellaswag|LogLoss|
|--|--|--|
|30B train|49.47|2.46|
|120B train|57.07|2.33|

MoE model
|1.5B of 10.4B (MoE)|Hellaswag|LogLoss|
|--|--|--|
|30B train|54.82|2.34|
|120B train|62.58|2.12|
