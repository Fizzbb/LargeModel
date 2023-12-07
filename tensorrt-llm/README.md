# Tensorrt-llm

## 1. latency benchmark
first token latency: set the output_len as 1 in input_output_len
https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/docs/source/performance.md#low-latency

## 2. build your own models
https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0/examples/gpt

## 3. slurm container example
We start by preparing an sbatch script called tensorrt_llm_run.sub. That script contains the following code (you must replace the <REPLACE ...> strings with your own values):
```
#!/bin/bash
#SBATCH -o logs/tensorrt_llm.out
#SBATCH -e logs/tensorrt_llm.error
#SBATCH -J <REPLACE WITH YOUR JOB's NAME>
#SBATCH -A <REPLACE WITH YOUR ACCOUNT's NAME>
#SBATCH -p <REPLACE WITH YOUR PARTITION's NAME>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

sudo nvidia-smi -lgc 1410,1410

srun --mpi=pmix \
     --container-image <image> \
     --container-mounts <path>:<path> \
     --container-workdir <path> \
     --output logs/tensorrt_llm_%t.out \
     --error logs/tensorrt_llm_%t.error python3 -u run.py --max_output_len=8 --engine_dir <engine_dir>
```
Then, submit the job using:

```sbatch tensorrt_llm_run.sub```
