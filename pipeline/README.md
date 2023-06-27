# Pipeline Parallelism

## 1. Pytorch Pipe
[Pytorch pipeline training reference](https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html)

[Tutorial Script](./torch-pipeline-transformer.py)

To demonstrate training large Transformer models using pipeline parallelism, we scale up the Transformer layers appropriately. We use an embedding dimension of 4096, hidden size of 4096, 16 attention heads and 12 total transformer layers (nn.TransformerEncoderLayer). This creates a model with ~1.4 billion parameters.

We need to initialize the RPC Framework since Pipe depends on the RPC framework via RRef which allows for future expansion to cross host pipelining. We need to initialize the RPC framework with only a single worker since we’re using a single process to drive multiple GPUs.

The pipeline is then initialized with 8 transformer layers on one GPU and 8 transformer layers on the other GPU (? why not 6, 6, a total of 12).

## 2. DeepSpeed Pipeline
```pip install deepspeed```

[Tutorial](https://www.deepspeed.ai/tutorials/pipeline/)

[bing bert](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/bing_bert)

[DeepSpeed Example](https://github.com/microsoft/DeepSpeedExamples/tree/master)

[AMD reference](https://cloudblogs.microsoft.com/opensource/2022/03/21/supporting-efficient-large-model-training-on-amd-instinct-gpus-with-deepspeed/)
[Intel reference]()
