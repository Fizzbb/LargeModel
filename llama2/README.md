## Download huggingface model
Prerequisitive
```
pip install huggingface_hub
huggingface-cli login
#fill in your huggingface token
```

### download use [snapshot_download](https://huggingface.co/docs/huggingface_hub/main/en/guides/download)
snapshot_download() downloads an entire repository at a given revision. It uses internally hf_hub_download() which means all downloaded files are also cached on your local disk. Downloads are made concurrently to speed-up the process.

To download a whole repository, just pass the repo_id and repo_type:

```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# Or from a dataset
snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

### hf version
``` pip install transformers accelerate```

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/home/aac/models")

model = AutoModelForCausalLM.from_pretrained(model, cache_dir="/home/aac/models", device_map="auto", torch_dtype=torch.float16)
```
The model json file is actually under the snapshot, so when to include the path it is like 
```
/home/aac/models/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9/
```

```
ls
config.json             model-00001-of-00002.safetensors  model.safetensors.index.json  tokenizer.json   tokenizer_config.json
generation_config.json  model-00002-of-00002.safetensors  special_tokens_map.json       tokenizer.model
```

### generic download
``` git clone https://huggingface.co/meta-llama/Llama-2-7b```
