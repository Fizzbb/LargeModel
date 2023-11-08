# Basics

## Interactive

**salloc**, let you get in the node and reserve for your use, after job is done, with exit command to leave the node and release resources

```
# -N to choose number of nodes to use
salloc --exclusive --mem=0 --gres=gpu:8 -p 1CN96C8G1H_4IB_MI250_Ubuntu22 -N 2
```
**srun**

## Batch run (submit to queue)
reference https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
```
sbatch -p 1CN96C8G1H_4IB_MI250_Ubuntu22 xxxx_sbatch.sh
```
Typical sbatch script structure
```
#!/bin/bash -x
#### part 1 #####
#SBATCH --nodes=32
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=open_clip
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition PARTITION_NAME

### part 2 ####
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

### part 3 ### same as the single node run command
srun torchrun \
--nnodes 4 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
/shared/examples/multinode_torchrun.py 50 10
```

# Examples
## 1 Multi-node rccl/nccl
https://github.com/amddcgpuce/AMDAcceleratorCloudGuides/blob/main/AACPlanoSlurmCluster/HowToGuides/How_To_Run_RCCL_Tests.md
