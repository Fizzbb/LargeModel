# LargeModel

## Fine-tune with single node
- https://blogs.oracle.com/research/post/oracle-first-to-finetune-gpt3-sized-ai-models-with-nvidia-a100-gpu
- llama2 https://github.com/facebookresearch/llama-recipes

## Model reference
- https://github.com/HabanaAI/Model-References

## Tracing
- https://github.com/pytorch/kineto

## Profiling
- Performance diagnosis toolkit https://arxiv.org/pdf/2205.02473.pdf
- nsys profile cli (2023.3) guide https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-options
- https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html#profiling_pytorch_pyprof
- https://github.com/NVIDIA/PyProf
- https://jingchaozhang.github.io/DLProf-Demo/

Trace analysis: https://github.com/facebookresearch/HolisticTraceAnalysis/tree/main/examples

- https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler
 
Rewrite
- https://github.com/adityaiitb/PyProf
- https://github.com/awwong1/torchprof
- https://github.com/cli99/flops-profiler

## Model Visulization 
- Netron https://github.com/lutzroeder/netron

## Training time, Flops estimation
- https://epochai.org/blog/estimating-training-compute
- https://github.com/cli99/flops-profiler/tree/main

## GPU benchmarks
- https://github.com/microsoft/superbenchmark
- https://github.com/te42kyfo/gpu-benches
```
git clone https://github.com/te42kyfo/gpu-benches.git
cd gpu-benches/gpu-stream/
/usr/local/cuda/bin/nvcc -o stream main.cu
./stream
```
