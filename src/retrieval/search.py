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


def run_keyword_search(
    chunks: list[dict],
    bm25: BM25Okapi,
    query: str,
) -> list[tuple[int, float]]:
    """Run BM25 keyword search and return ranked chunk indices."""
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:TOP_K]

    return [
        (int(index), float(scores[index]))
        for index in ranked_indices
    ]


def run_dense_search(
    chunks: list[dict],
    model: SentenceTransformer,
    client: QdrantClient,
    query: str,
) -> list[tuple[int, float]]:
    """Run dense semantic search using BGE embeddings and Qdrant."""
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    ).points

    ranked_results = []

    for result in results:
        chunk_id = result.payload["chunk_id"]

        for index, chunk in enumerate(chunks):
            if chunk["id"] == chunk_id:
                ranked_results.append(
                    (index, float(result.score))
                )
                break

    return ranked_results


def run_hybrid_search(
    chunks: list[dict],
    bm25: BM25Okapi,
    model: SentenceTransformer,
    client: QdrantClient,
    query: str,
) -> list[tuple[int, float, float, float]]:
    """
    Run hybrid search using normalized dense and BM25 scores.

    Returns:
        (chunk_index, hybrid_score, dense_score, keyword_score)
    """

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
                dense_scores[index] = float(result.score)
                break

    query_tokens = tokenize(query)

    keyword_scores = bm25.get_scores(query_tokens).tolist()

    normalized_dense = normalize_scores(dense_scores)
    normalized_keyword = normalize_scores(keyword_scores)

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

    return [
        (
            index,
            hybrid_scores[index],
            dense_scores[index],
            keyword_scores[index],
        )
        for index in ranked_indices
    ]


def print_keyword_results(
    chunks: list[dict],
    results: list[tuple[int, float]],
) -> None:
    """Display BM25 search results."""
    print("\n" + "=" * 60)
    print("BM25 / KEYWORD SEARCH")
    print("=" * 60)

    for rank, (index, score) in enumerate(results, start=1):
        chunk = chunks[index]

        print(f"\nRank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk: {chunk['id']}")
        print(f"\nText:\n{chunk['text'][:500]}")


def print_dense_results(
    chunks: list[dict],
    results: list[tuple[int, float]],
) -> None:
    """Display dense semantic search results."""
    print("\n" + "=" * 60)
    print("DENSE / SEMANTIC SEARCH")
    print("=" * 60)

    for rank, (index, score) in enumerate(results, start=1):
        chunk = chunks[index]

        print(f"\nRank: {rank}")
        print(f"Cosine Score: {score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk: {chunk['id']}")
        print(f"\nText:\n{chunk['text'][:500]}")


def print_hybrid_results(
    chunks: list[dict],
    results: list[tuple[int, float, float, float]],
) -> None:
    """Display hybrid search results."""
    print("\n" + "=" * 60)
    print("HYBRID SEARCH")
    print("=" * 60)

    print(
        f"Weighting: "
        f"{DENSE_WEIGHT:.0%} Dense / "
        f"{KEYWORD_WEIGHT:.0%} Keyword"
    )

    for rank, (
        index,
        hybrid_score,
        dense_score,
        keyword_score,
    ) in enumerate(results, start=1):

        chunk = chunks[index]

        print(f"\nRank: {rank}")
        print(f"Hybrid Score: {hybrid_score:.4f}")
        print(f"Dense Score: {dense_score:.4f}")
        print(f"Keyword Score: {keyword_score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk: {chunk['id']}")
        print(f"\nText:\n{chunk['text'][:500]}")


def main() -> None:
    print("=" * 60)
    print("SEMANTIC SEARCH ENGINE")
    print("=" * 60)

    print("\nLoading corpus...")
    chunks = load_chunks()

    print("Loading BM25 index...")
    documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    bm25 = BM25Okapi(documents)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ").strip()

        if query.lower() == "exit":
            print("\nExiting search engine.")
            break

        if not query:
            print("Please enter a search query.")
            continue

        print(f"\nSearching for: {query}")

        keyword_results = run_keyword_search(
            chunks,
            bm25,
            query,
        )

        dense_results = run_dense_search(
            chunks,
            model,
            client,
            query,
        )

        hybrid_results = run_hybrid_search(
            chunks,
            bm25,
            model,
            client,
            query,
        )

        print_keyword_results(
            chunks,
            keyword_results,
        )

        print_dense_results(
            chunks,
            dense_results,
        )

        print_hybrid_results(
            chunks,
            hybrid_results,
        )

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()