
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"

print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager"
)

# Sample sentence
text = "The student submitted her assignment because she completed it."

inputs = tokenizer(
    text,
    return_tensors="pt"
)

# Get model outputs and attention weights
with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True
    )

# Select the last Transformer layer
# Shape: batch, heads, sequence, sequence
attention = outputs.attentions[-1][0]

# Select the first attention head
attention_head = attention[0].numpy()

# Get token labels
tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

# Display attention heatmap
plt.figure(figsize=(10, 8))

plt.imshow(attention_head, cmap="viridis")

plt.xticks(
    range(len(tokens)),
    tokens,
    rotation=90
)

plt.yticks(
    range(len(tokens)),
    tokens
)

plt.xlabel("Tokens being attended to")
plt.ylabel("Current token")
plt.title("Transformer Self-Attention - One Attention Head")

plt.colorbar(label="Attention weight")
plt.tight_layout()
plt.show()
