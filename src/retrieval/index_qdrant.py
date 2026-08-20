import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/chunks.json"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_search"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


def load_chunks() -> list[dict]:
    """Load processed chunks from disk."""
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print("Connecting to Qdrant...")

    client = QdrantClient(url=QDRANT_URL)

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Vector dimension: {embeddings.shape[1]}")

    print(f"\nCreating Qdrant collection: {COLLECTION_NAME}")

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for point_index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        point = PointStruct(
            id=point_index,
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            },
        )

        points.append(point)

    print(f"Uploading {len(points)} points to Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    collection_info = client.get_collection(COLLECTION_NAME)

    print("\nIndexing complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points stored: {collection_info.points_count}")
    print(f"Vector size: {VECTOR_SIZE}")
    print("Distance: COSINE")


if __name__ == "__main__":
    main()