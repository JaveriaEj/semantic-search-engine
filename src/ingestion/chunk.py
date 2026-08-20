import json
import os
import re


INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.json")

CHUNK_SIZE = 500
OVERLAP = 50


def clean_text(text: str) -> str:
    """Clean unnecessary whitespace from document text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def process_document(filename: str) -> list[dict]:
    """Read one document and convert it into chunks."""
    input_path = os.path.join(INPUT_DIR, filename)

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = clean_text(text)

    chunks = create_chunks(
        text,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )

    source = os.path.splitext(filename)[0]

    records = []

    for index, chunk in enumerate(chunks):
        record = {
            "id": f"{source}_{index}",
            "text": chunk,
            "source": source,
            "chunk_index": index,
        }

        records.append(record)

    return records


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []

    filenames = sorted(
        filename
        for filename in os.listdir(INPUT_DIR)
        if filename.endswith(".txt")
    )

    for filename in filenames:
        records = process_document(filename)

        all_chunks.extend(records)

        print(f"Processed: {filename}")
        print(f"Chunks: {len(records)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_chunks, file, indent=2, ensure_ascii=False)

    print("\nChunking complete.")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()