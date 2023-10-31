## Command to run
```
CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py --mini-batch-size=2048 --debug-mode --test-mini-batch-size=16384 --test-num-workers=0 --num-batches=1000 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time --enable-profiling --use-gpu > model1_GPU_PT_1.log
```

## Command with dlprof

```
CUDA_VISIBLE_DEVICES=0 dlprof --mode=pytorch  --reports=all --iter_start 50 --iter_stop 60 --output_path=./log3 python dlrm_pt_dlprof.py --mini-batch-size=2048 --debug-mode --test-mini-batch-size=16384 --test-num-workers=0 --num-batches=100 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time --use-gpu
```
