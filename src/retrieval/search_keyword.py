import json

from rank_bm25 import BM25Okapi


INPUT_FILE = "data/processed/chunks.json"
TOP_K = 3


def load_chunks() -> list[dict]:
    """Load processed chunks from disk."""
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return text.lower().split()


def main() -> None:
    chunks = load_chunks()

    documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(documents)

    query = input("\nEnter your search query: ")

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:TOP_K]

    print("\n" + "=" * 60)
    print("KEYWORD SEARCH RESULTS")
    print("=" * 60)

    for rank, index in enumerate(ranked_indices, start=1):
        chunk = chunks[index]

        print(f"\nRank: {rank}")
        print(f"Score: {scores[index]:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk: {chunk['id']}")
        print(f"\nText:\n{chunk['text'][:500]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()