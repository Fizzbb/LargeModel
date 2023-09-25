# Basics

**salloc**, let you get in the node and reserve for your use, after job is done, with exit command to leave the node and release resources

```
# -N to choose number of nodes to use
salloc --exclusive --mem=0 --gres=gpu:8 -p 1CN96C8G1H_4IB_MI250_Ubuntu22 -N 2
```

# Examples
## 1 Multi-node rccl/nccl
https://github.com/amddcgpuce/AMDAcceleratorCloudGuides/blob/main/AACPlanoSlurmCluster/HowToGuides/How_To_Run_RCCL_Tests.md
