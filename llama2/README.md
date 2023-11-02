## Download huggingface model
```
pip install huggingface_hub
huggingface-cli login
#fill in your huggingface token
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

### generic download
``` git clone https://huggingface.co/meta-llama/Llama-2-7b```
