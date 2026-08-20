import json

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/chunks.json"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_search"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 3

DENSE_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.5


def load_chunks() -> list[dict]:
    """Load processed chunks from disk."""
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return text.lower().split()


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to the range 0 to 1."""
    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def main() -> None:
    print("Loading corpus...")

    chunks = load_chunks()

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Qdrant...")

    client = QdrantClient(url=QDRANT_URL)

    documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(documents)

    query = input("\nEnter your search query: ")

    # -----------------------------
    # Dense search
    # -----------------------------

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=len(chunks),
        with_payload=True,
        with_vectors=False,
    ).points

    dense_scores = [0.0] * len(chunks)

    for result in dense_results:
        chunk_id = result.payload["chunk_id"]

        for index, chunk in enumerate(chunks):
            if chunk["id"] == chunk_id:
                dense_scores[index] = result.score
                break

    # -----------------------------
    # Keyword search
    # -----------------------------

    query_tokens = tokenize(query)

    keyword_scores = bm25.get_scores(query_tokens).tolist()

    # -----------------------------
    # Normalize scores
    # -----------------------------

    normalized_dense = normalize_scores(dense_scores)
    normalized_keyword = normalize_scores(keyword_scores)

    # -----------------------------
    # Hybrid scoring
    # -----------------------------

    hybrid_scores = []

    for dense_score, keyword_score in zip(
        normalized_dense,
        normalized_keyword,
    ):
        hybrid_score = (
            DENSE_WEIGHT * dense_score
            + KEYWORD_WEIGHT * keyword_score
        )

        hybrid_scores.append(hybrid_score)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: hybrid_scores[index],
        reverse=True,
    )[:TOP_K]

    # -----------------------------
    # Display results
    # -----------------------------

    print("\n" + "=" * 60)
    print("HYBRID SEARCH RESULTS")
    print("=" * 60)

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[index]

        print(f"\nRank: {rank}")
        print(f"Hybrid Score: {hybrid_scores[index]:.4f}")
        print(f"Dense Score: {dense_scores[index]:.4f}")
        print(f"Keyword Score: {keyword_scores[index]:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk: {chunk['id']}")
        print(f"\nText:\n{chunk['text'][:500]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()