import os
import requests
from bs4 import BeautifulSoup


ARTICLES = [
    {
        "name": "hnsw",
        "url": "https://www.pinecone.io/learn/series/rag/hnsw/",
    },
    {
        "name": "hybrid_search",
        "url": "https://www.pinecone.io/learn/hybrid-search-intro/",
    },
    {
        "name": "semantic_search",
        "url": "https://www.pinecone.io/learn/search-with-pinecone/",
    },
    {
        "name": "rag",
        "url": "https://www.pinecone.io/learn/retrieval-augmented-generation/",
    },
    {
        "name": "embedding_models",
        "url": "https://www.pinecone.io/learn/series/rag/embedding-models-rundown/",
    },
]

OUTPUT_DIR = "data/raw"

STOP_MARKERS = [
    "Get started today",
    "References",
    "Share:",
    "Recommended for you",
    "Further Reading",
]

MINIMUM_CONTENT_LINES = 30


def extract_article_text(html: str) -> str:
    """Extract and clean the main article content."""
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "nav", "footer", "aside"]):
        element.decompose()

    content = soup.find("article")

    if content is None:
        content = soup.find("main")

    if content is None:
        content = soup.body

    if content is None:
        content = soup

    text = content.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cleaned_lines = []

    for line in lines:
        is_stop_marker = any(
            marker in line for marker in STOP_MARKERS
        )

        if is_stop_marker and len(cleaned_lines) >= MINIMUM_CONTENT_LINES:
            break

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def download_article(url: str, output_file: str) -> None:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    response.raise_for_status()

    article_text = extract_article_text(response.text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(article_text)

    print(f"Saved: {output_file}")
    print(f"Characters: {len(article_text):,}")


def main() -> None:
    for article in ARTICLES:
        output_file = os.path.join(
            OUTPUT_DIR,
            f"{article['name']}.txt",
        )

        try:
            download_article(article["url"], output_file)
        except requests.RequestException as error:
            print(f"Failed: {article['name']}")
            print(f"Error: {error}")

    print("\nCorpus download complete.")


if __name__ == "__main__":
    main()