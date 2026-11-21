
from transformers import AutoTokenizer

MODEL_ID = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

text = "Artificial Intelligence is transforming agriculture."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Decoded text:", tokenizer.decode(token_ids))
