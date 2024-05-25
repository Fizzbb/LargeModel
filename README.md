# LargeModel

## Transformer
- Illustration: https://jalammar.github.io/illustrated-transformer/
- Model Flops calculation: https://zhuanlan.zhihu.com/p/624740065
- Model/token/communication estimation: https://www.53ai.com/news/qianyanjishu/303.html
- collective communication: https://zhuanlan.zhihu.com/p/435438871
- GPU capability/bottleneck: https://mp.weixin.qq.com/s/S7lxmi_Q_Uq23mtMus4KSQ 

## Training from scratch
- https://www.youtube.com/watch?v=ZLbVdvOoTKM

## Training framework performance 
- Megatron analysis: https://www.high-flyer.cn/blog/model_parallel-1/inidex/
  
## Fine-tune with single node
- https://blogs.oracle.com/research/post/oracle-first-to-finetune-gpt3-sized-ai-models-with-nvidia-a100-gpu
- llama2 https://github.com/facebookresearch/llama-recipes

## Inference explaination
- https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
- https://jalammar.github.io/illustrated-gpt2/#part-1-got-and-language-modeling

## Performance projection
- https://mp.weixin.qq.com/s/ftF3YRXPZ5mjVfqzGCDNYQ

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
- https://bbycroft.net/llm
  
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

## GPU foundamentals 
- How GEMM works https://siboehm.com/articles/22/CUDA-MMM

## Compilation
- TVM to custom ML hardware https://www.youtube.com/watch?v=FBdW1gJGx0M
