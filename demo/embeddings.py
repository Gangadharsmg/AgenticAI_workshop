
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

texts = [
    "Tomato plants need sufficient nutrients.",
    "Tomato crops require proper soil nutrition.",
    "Deep learning is used for image classification."
]

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts
)

for text, embedding in zip(texts, response.embeddings):
    print("\nText:", text)
    print("Embedding dimensions:", len(embedding.values))
    print("First five values:", embedding.values[:5])
