from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# must add, otherwise runs on CPU
model.to("cuda")
encoded_input.to("cuda")

for i in range(100):  #make program run longer to track GPU utils
    output = model(**encoded_input)
