import json
import os

from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/chunks.json"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks() -> list[dict]:
    """Load chunk records from the processed corpus."""
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks()

    texts = [chunk["text"] for chunk in chunks]

    print(f"Chunks loaded: {len(texts)}")
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    print("\nEmbedding generation complete.")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("\nFirst embedding:")
    print(embeddings[0])

    print("\nFirst chunk:")
    print(chunks[0]["text"][:300])


if __name__ == "__main__":
    main()