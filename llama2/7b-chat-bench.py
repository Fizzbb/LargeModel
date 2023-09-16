from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time
from datetime import datetime

model = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda:0"
NEW_TOKENS_LEN = [32, 64]

tokenizer = AutoTokenizer.from_pretrained(model)

model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)

prompt = "The basic idea of a Transformer model is"

BS = BS_FOR_TEST = 1
prompts = [prompt] * BS
tokenizer_tik = time.time()
input_ids = tokenizer(prompts, return_tensors="pt").to(device).input_ids
# input_ids = torch.randint(100, 4000, (BS, 2047)).to(device) # if we need random inputs
ilen = input_ids.shape[1] # assume all inputs are of same length
tokenizer_tok = time.time()
stime = "%.3f" % (tokenizer_tok-tokenizer_tik)
print(f"{datetime.now()}:: {stime} seconds taken for tokenizer {input_ids.shape}.", flush=True)

with torch.inference_mode():
    with torch.no_grad():#, torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        # Do 2 warmups and then measure the perf
        output_ids = model.generate(input_ids, max_new_tokens=NEW_TOKENS_LEN[0])
        print(f"{datetime.now()}:: First warmup done...", flush=True)
        output_ids = model.generate(input_ids, max_new_tokens=NEW_TOKENS_LEN[0])
        print(f"{datetime.now()}:: Second warmup done...", flush=True)
        tik = time.time()
        output_ids = model.generate(input_ids, max_new_tokens=NEW_TOKENS_LEN[0])
        tok = time.time()

print(output_ids.shape)
input_texts = tokenizer.batch_decode(output_ids[:, :ilen], skip_special_tokens=True, clean_up_tokenization_spaces=False)
output_texts = tokenizer.batch_decode(output_ids[:, ilen:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Input: ", input_texts[0])
print("Q00: ", output_texts[0])
stime = "%.3f" % (tok-tik)
stoken_rate = "%.3f" % (BS*NEW_TOKENS_LEN[0]/(tok - tik))
print(f"{stime} seconds taken for generating the response ({stoken_rate} tokens/sec).", flush=True)

