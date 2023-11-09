## Command to run, synthetic data
```
CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py --mini-batch-size=2048 --debug-mode --test-mini-batch-size=16384 --test-num-workers=0 --num-batches=1000 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time --enable-profiling --use-gpu > model1_GPU_PT_1.log
```

## run a larger model with criteo-7days data

CUDA_VISIBLE_DEVICES=0
python dlrm_s_pytorch.py --use-gpu --arch-sparse-feature-size=128 --arch-embedding-size=128 --arch-mlp-top="1024-1024-512-256-1" --arch-mlp-bot="512-256-128" --data-generation=dataset --data-set=kaggle --raw-data-file=/root/criteo-7days/train.txt --processed-data-file=/root/criteo-7days/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=12800 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log


## Command with dlprof

```
CUDA_VISIBLE_DEVICES=0 dlprof --mode=pytorch  --reports=all --iter_start 50 --iter_stop 60 --output_path=./log3 python dlrm_pt_dlprof.py --mini-batch-size=2048 --debug-mode --test-mini-batch-size=16384 --test-num-workers=0 --num-batches=100 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time --use-gpu
```
