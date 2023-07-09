from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# must add, otherwise runs on CPU
model.to("cuda")
encoded_input.to("cuda")

for i in range(100):  #make program run longer to track GPU utils
    output = model(**encoded_input)
    print(output)
