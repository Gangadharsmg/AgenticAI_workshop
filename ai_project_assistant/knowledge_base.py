
import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "ai_project_knowledge"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBEDDING_BATCH_SIZE = 16
TOP_K = 3


def create_gemini_client():
    """Create a Gemini API client."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Configure it in your .env file."
        )

    return genai.Client(api_key=api_key)


def create_chroma_collection():
    """Create or retrieve the persistent ChromaDB collection."""

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    return chroma_client, collection


def load_documents():
    """Load all text files from the data directory."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    documents = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": file_path.name,
                "text": text
            })

    if not documents:
        raise RuntimeError(
            "No non-empty .txt documents found in the data directory."
        )

    return documents


def split_text_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Prefer a word or paragraph boundary when possible.
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(" ", start + chunk_size // 2, end)
            )

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def generate_embeddings(client, texts, task_type):
    """Generate Gemini embeddings for a list of texts."""

    all_embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):

        batch = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type
            )
        )

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no embeddings."
            )

        if len(response.embeddings) != len(batch):
            raise RuntimeError(
                "Embedding count does not match input count."
            )

        for embedding in response.embeddings:
            if not embedding.values:
                raise RuntimeError(
                    "An empty embedding was returned."
                )

            all_embeddings.append(embedding.values)

    return all_embeddings


def build_knowledge_base(client, collection):
    """Load, chunk, embed and store documents in ChromaDB."""

    documents = load_documents()

    all_chunks = []
    all_metadatas = []
    all_ids = []

    print("\nLoading and splitting documents...\n")

    for document in documents:

        source = document["source"]
        text = document["text"]

        chunks = split_text_into_chunks(text)

        print(f"{source}: {len(chunks)} chunks")

        # Remove old chunks from this source before re-indexing.
        # Other source documents remain untouched.
        if collection.count() > 0:
            collection.delete(
                where={"source": source}
            )

        for index, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            all_ids.append(
                f"{Path(source).stem}-{index}"
            )

            all_metadatas.append({
                "source": source,
                "chunk_index": index
            })

    if not all_chunks:
        raise RuntimeError("No document chunks were created.")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings with Gemini...\n")

    embeddings = generate_embeddings(
        client,
        all_chunks,
        task_type="RETRIEVAL_DOCUMENT"
    )

    print("Storing vectors in ChromaDB...\n")

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )

    print("Knowledge base created successfully!")
    print(f"Total records in ChromaDB: {collection.count()}")


def search_knowledge_base(client, collection, question, top_k=TOP_K):
    """Retrieve relevant document chunks for a question."""

    if collection.count() == 0:
        raise RuntimeError(
            "Knowledge base is empty. Run the build command first."
        )

    query_embedding = generate_embeddings(
        client,
        [question],
        task_type="RETRIEVAL_QUERY"
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results


def display_search_results(results):
    """Display retrieved chunks and their metadata."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n" + "=" * 60)
    print("           RETRIEVED KNOWLEDGE")
    print("=" * 60)

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\nResult {index}")
        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"\nContent:\n{document}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build and search the AI Project knowledge base."
    )

    parser.add_argument(
        "command",
        choices=["build", "search"],
        help="Choose build or search."
    )

    parser.add_argument(
        "--query",
        help="Question to search for."
    )

    args = parser.parse_args()

    try:
        client = create_gemini_client()
        _, collection = create_chroma_collection()

        if args.command == "build":
            build_knowledge_base(client, collection)

        elif args.command == "search":

            question = args.query

            if not question:
                question = input(
                    "\nEnter your search question: "
                ).strip()

            if not question:
                raise ValueError("Please enter a search question.")

            results = search_knowledge_base(
                client,
                collection,
                question
            )

            display_search_results(results)

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
