# Introduction

## HIP
HIP: a GPU programming language, a wrapper API can be compiled to either CUDA or ROCm devices.

HIPCC (hcc): compiler

Programming language --> compiler --> hardware specific assembly

AMD Intro : https://www.youtube.com/playlist?list=PLx15eYqzJifehAxhWRD6T35GZwAqM9IK4

## TVM
ML Model --> format change --> optimization --> compile --> llvm target (for cpu run)

TVM: a more capable compiler for production ML deployments

Each operator in the graph must be optimized for acceleration on hardware target

Supported hardware backend, https://tvm.apache.org/docs/tutorial/relay_quick_start.html

TVM pythonAPI example : https://www.youtube.com/watch?v=A2WlUuAgT3w

https://www.youtube.com/watch?v=UGHk402K-Oo


model visualization
https://github.com/lutzroeder/netron
