from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import AutoConfig
import torch

print('Pytorch version\t:', torch.__version__)
print('CUDA version\t:', torch.version.cuda)

import inspect
from collections import defaultdict
import pandas as pd
from torch.utils import benchmark 

pd.options.display.precision = 3

def var_dict(*args):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return dict([(name, val) for name, val in callers_local_vars if val is arg][0] 
                for arg in args)

def walltime(stmt, arg_dict, duration=3):
    return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(
        min_run_time=duration).median


config = AutoConfig.from_pretrained("gpt2-medium")
layer = GPT2Block(config, layer_idx=0).half().cuda()

def layer_benchmark(layer, hidden_size, seq_lens, batch_sizes, cross_attention=False):
    h = hidden_size
    results = defaultdict(lambda: {})    
    encoder_state = 'encoder_hidden_states=X' if cross_attention else ''
    for s in seq_lens:
        for b in batch_sizes:            
            ffn = 16*b*s*h*h / 1e12  # TFLOPS for the Feed-Forward Network
            atten = (4*b*h*s*s + 8*b*s*h*h) / 1e12  # TFLOPS for attention            
            forward = ffn + (2 if cross_attention else 1) * atten
            
            X = torch.randn(b, s, h).half().cuda()
            results[f'batch={b}'][f'fwd seq_len={s}'] = forward / walltime(
                f'layer(X, {encoder_state})', var_dict(layer, X))
            results[f'batch={b}'][f'fwd+bwd seq_len={s}'] = 3 * forward / walltime(
                f'layer(X, {encoder_state})[0].sum().backward()', var_dict(layer, X))            
    return pd.DataFrame(results)

res = layer_benchmark(layer, config.n_embd, [512, 1024], [2, 4, 8, 16, 32, 64])



print(res)

