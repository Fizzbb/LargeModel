# Methodologies

## [Roofline model](https://en.wikipedia.org/wiki/Roofline_model)

The Roofline model is then obtained by plotting the performance(FLOPS) versus the arithmetic intensity(FLOPs/bytes).

## Figure of Merits (FOM)
Figure of Merit Regard less of the types of the benchmarks, a relative metric,i.e.,figure of merit(FOM),is often used in procurement. In this study,it is defined as follow
https://www.sciencedirect.com/science/article/pii/S2772485921000053

## MFU (Model Flop Utilization)
1. https://arxiv.org/pdf/2205.05198.pdf
2. https://github.com/NVIDIA/Megatron-LM


## Benchmarking

1. https://github.com/mli/transformers-benchmarks

2. Tensor core utilization to represent hardware flops utilization (HFU)
```dcgmi dmon -e 203,1001,1002,1003,1004,1005,1009,1010,1011,1012 -i 0```
