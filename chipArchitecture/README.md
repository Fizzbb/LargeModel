# Accelerator architecture comparison

https://khairy2011.medium.com/tpu-vs-gpu-vs-cerebras-vs-graphcore-a-fair-comparison-between-ml-hardware-3f5a19d89e38

https://drive.google.com/file/d/1bElFjVNoQEoLIAzQFxW3-qsBWmI6KXbj/view?pli=1

- Systolic Array (TPU) 
- SIMD + TC (Nvidia GPU) 
- MIMD (Cerebras, GraphCore, Tenstorrent)

## Comparison Metrics
However, MLPerf only considers the raw performance (i.e., training time) as a key measurement and does not take into account other important metrics, such as 
- compute efficiency (hardware utilization)
- power efficiency (performance/watt)
- area efficiency (performance/rack)
- cost efficiency (performance/dollar)

## 1. GPU, data parallelism

## 2. Dataflow, model parallelism

The model parallelism exploits the intra-layer (GEMM operation) and inter-layer (pipeline) parallelism by running multiple layers in parallel and transferring data between layers in a dataflow manner.
