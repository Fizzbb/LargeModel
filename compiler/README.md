# Introduction

https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html

https://hamel.dev/notes/serving/

## HIP
HIP: a GPU programming language, a wrapper API can be compiled to either CUDA or ROCm devices.

HIPCC (hcc): compiler

libraries: HCCL, OpenMI ...

Programming language --> compiler --> hardware specific assembly

AMD Intro : https://www.youtube.com/playlist?list=PLx15eYqzJifehAxhWRD6T35GZwAqM9IK4


## Triton (OpenAI)
Triton: a GPU programming lanuage, allow user to write performant GPU kernels

Triton Compiler --> CUDA and RoCm backend

https://hamel.dev/notes/serving/

## PopLar (GraphCore)
https://www.graphcore.ai/hubfs/Cambrian%20AI%20white%20paper_Graphcores%20AI%20software%20stack%20is%20now%20customer%20driven.pdf

Graphcore is the only AI company that has released the source code of their ML
framework backend implementations.(POPART, POPDIST)


## TVM (serving)
ML Model --> format change --> optimization --> compile --> llvm target (for cpu run)

TVM: a more capable compiler for production ML deployments

Each operator in the graph must be optimized for acceleration on hardware target

Supported hardware backend, https://tvm.apache.org/docs/tutorial/relay_quick_start.html

TVM pythonAPI example : https://www.youtube.com/watch?v=A2WlUuAgT3w

https://www.youtube.com/watch?v=UGHk402K-Oo



model visualization
https://github.com/lutzroeder/netron
