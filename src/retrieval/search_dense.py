from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_search"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 3


def main() -> None:
    print("Connecting to Qdrant...")

    client = QdrantClient(url=QDRANT_URL)

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    query = input("\nEnter your search query: ")

    print("\nGenerating query embedding...")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    print("Searching Qdrant...")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    ).points

    print("\n" + "=" * 60)
    print("DENSE SEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.payload['source']}")
        print(f"Chunk: {result.payload['chunk_id']}")
        print(f"\nText:\n{result.payload['text'][:500]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()