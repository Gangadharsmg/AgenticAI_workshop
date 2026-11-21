
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "ai_project_knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """Retrieve relevant knowledge chunks from the existing ChromaDB collection."""

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    query_embedding = response.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No relevant information was found in the knowledge base."

    return "\n\n".join(
        f"Knowledge {index}:\n{document}"
        for index, document in enumerate(documents, start=1)
    )
