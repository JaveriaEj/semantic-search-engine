
import json

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = "data/processed/chunks.json"
QUERIES_FILE = "src/evaluation/queries.json"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_search"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 3


# Hybrid configurations to test.
HYBRID_CONFIGS = [
    {
        "name": "Hybrid 25D/75K",
        "dense_weight": 0.25,
        "keyword_weight": 0.75,
    },
    {
        "name": "Hybrid 50D/50K",
        "dense_weight": 0.50,
        "keyword_weight": 0.50,
    },
    {
        "name": "Hybrid 75D/25K",
        "dense_weight": 0.75,
        "keyword_weight": 0.25,
    },
]


def load_json(path: str):
    """Load JSON data from disk."""
    with open(path, "r", encoding="utf-8") as file:
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


def precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Calculate Precision@K."""
    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        chunk_id in relevant
        for chunk_id in retrieved_at_k
    )

    return relevant_count / k


def recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Calculate Recall@K."""
    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        chunk_id in relevant
        for chunk_id in retrieved_at_k
    )

    return relevant_count / len(relevant)


def reciprocal_rank(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def dense_search(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
) -> list[str]:
    """Return chunk IDs ranked by dense similarity."""

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

    return [
        result.payload["chunk_id"]
        for result in results
    ]


def keyword_search(
    query: str,
    chunks: list[dict],
    bm25: BM25Okapi,
) -> list[str]:
    """Return chunk IDs ranked by BM25."""

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:TOP_K]

    return [
        chunks[index]["id"]
        for index in ranked_indices
    ]


def hybrid_search(
    query: str,
    chunks: list[dict],
    bm25: BM25Okapi,
    model: SentenceTransformer,
    client: QdrantClient,
    dense_weight: float,
    keyword_weight: float,
) -> list[str]:
    """Return chunk IDs ranked by hybrid score."""

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

    query_tokens = tokenize(query)

    keyword_scores = bm25.get_scores(query_tokens).tolist()

    normalized_dense = normalize_scores(dense_scores)
    normalized_keyword = normalize_scores(keyword_scores)

    hybrid_scores = []

    for dense_score, keyword_score in zip(
        normalized_dense,
        normalized_keyword,
    ):
        score = (
            dense_weight * dense_score
            + keyword_weight * keyword_score
        )

        hybrid_scores.append(score)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: hybrid_scores[index],
        reverse=True,
    )[:TOP_K]

    return [
        chunks[index]["id"]
        for index in ranked_indices
    ]


def evaluate_method(
    method_name: str,
    results: list[list[str]],
    queries: list[dict],
) -> dict:
    """Calculate average evaluation metrics."""

    precisions = []
    recalls = []
    reciprocal_ranks = []

    for retrieved, query_data in zip(results, queries):
        relevant = query_data["relevant_chunks"]

        precisions.append(
            precision_at_k(
                retrieved,
                relevant,
                TOP_K,
            )
        )

        recalls.append(
            recall_at_k(
                retrieved,
                relevant,
                TOP_K,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved,
                relevant,
            )
        )

    return {
        "method": method_name,
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def main() -> None:
    print("Loading corpus...")

    chunks = load_json(CHUNKS_FILE)

    print("Loading evaluation queries...")

    queries = load_json(QUERIES_FILE)

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Qdrant...")

    client = QdrantClient(url=QDRANT_URL)

    documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(documents)

    keyword_results = []
    dense_results = []

    hybrid_results = {
        config["name"]: []
        for config in HYBRID_CONFIGS
    }

    print("\nRunning evaluation...")

    for query_data in queries:
        query = query_data["query"]

        print(f"\nQuery: {query}")

        keyword = keyword_search(
            query,
            chunks,
            bm25,
        )

        dense = dense_search(
            query,
            model,
            client,
        )

        keyword_results.append(keyword)
        dense_results.append(dense)

        print(f"Keyword: {keyword}")
        print(f"Dense:   {dense}")

        for config in HYBRID_CONFIGS:
            hybrid = hybrid_search(
                query,
                chunks,
                bm25,
                model,
                client,
                config["dense_weight"],
                config["keyword_weight"],
            )

            hybrid_results[config["name"]].append(hybrid)

            print(f"{config['name']}: {hybrid}")

    keyword_metrics = evaluate_method(
        "Keyword",
        keyword_results,
        queries,
    )

    dense_metrics = evaluate_method(
        "Dense",
        dense_results,
        queries,
    )

    hybrid_metrics = []

    for config in HYBRID_CONFIGS:
        metrics = evaluate_method(
            config["name"],
            hybrid_results[config["name"]],
            queries,
        )

        hybrid_metrics.append(metrics)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    all_metrics = [
        keyword_metrics,
        dense_metrics,
        *hybrid_metrics,
    ]

    for metrics in all_metrics:
        print(f"\n{metrics['method']}")
        print(f"Precision@{TOP_K}: {metrics['precision']:.4f}")
        print(f"Recall@{TOP_K}:    {metrics['recall']:.4f}")
        print(f"MRR:               {metrics['mrr']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
