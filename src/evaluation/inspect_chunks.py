import json


INPUT_FILE = "data/processed/chunks.json"


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("=" * 80)
    print("CORPUS CHUNKS")
    print("=" * 80)

    for chunk in chunks:
        print("\n" + "-" * 80)
        print(f"ID: {chunk['id']}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print("-" * 80)
        print(chunk["text"])


if __name__ == "__main__":
    main()