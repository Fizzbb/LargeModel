# Reference 

- Model list
https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/samples

- NV DLRM data prep, training, inference example
https://developer.nvidia.com/blog/optimizing-dlrm-on-nvidia-gpus/

- DLRM deep dive
https://medium.com/swlh/deep-learning-recommendation-models-dlrm-a-deep-dive-f38a95f47c2c

- Embedding scaling/caching
https://www.hpc-ai.tech/blog/embedding-training-with-1-gpu-memory-and-10-times-less-budget-an-open-source-solution-for

- Embedding table size (vocabulary x embed_dim) x 4 (if use fp32)

The popular public dataset Criteo 1TB contains around one billion feature IDs. If embedding_dims is set to 128, then it requires 512 GB storage to accommodate the Embedding parameters. Moreover, if using the Adam optimizer, the required storage will spike to 1536 GB because of the need to store the extra status variables:m and v.


- Profiling with NV's pyProf
https://www.adityaagrawal.net/blog/dnn/dlrm  https://github.com/adityaiitb/dlperf/tree/master/Recommendation/DLRM
