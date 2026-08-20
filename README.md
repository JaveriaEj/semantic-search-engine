# Semantic Search Engine

A retrieval system that compares **keyword search, dense semantic search, and hybrid search** over a real AI/ML technical corpus.

The project investigates how different retrieval strategies behave as the corpus becomes larger and more diverse.

---

## Project Goal

The goal is to build and evaluate three retrieval approaches:

1. **Keyword retrieval** using BM25
2. **Dense semantic retrieval** using BGE embeddings and Qdrant
3. **Hybrid retrieval** combining BM25 and dense similarity scores

The systems are evaluated using a manually labeled query set and the metrics:

- Precision@3
- Recall@3
- MRR

### Research Question

> How do keyword, dense semantic, and hybrid retrieval strategies compare when retrieving relevant technical documents from an AI/ML corpus?

---

# Architecture

```text
                    Documents
                        │
                        ▼
                    Chunking
                        │
                        ▼
                   Embeddings
                        │
                        ▼
                    Qdrant
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Dense Retrieval         BM25 Retrieval
       BGE + Qdrant            Keyword Search
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                 Hybrid Retrieval
                        │
                        ▼
                    Evaluation
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Precision@3    Recall@3         MRR
```

---

# Project Structure

```text
semantic-search-engine/
│
├── data/
│   ├── raw/
│   │   ├── bm25.txt
│   │   ├── chunking.txt
│   │   ├── embedding_models.txt
│   │   ├── hnsw.txt
│   │   ├── hybrid_search.txt
│   │   ├── rag.txt
│   │   ├── reranking.txt
│   │   ├── semantic_search.txt
│   │   ├── vector_database.txt
│   │   └── vector_similarity.txt
│   │
│   └── processed/
│       └── chunks.json
│
├── src/
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── download.py
│   │   └── chunk.py
│   │
│   ├── embeddings/
│   │   └── embed.py
│   │
│   ├── retrieval/
│   │   ├── index_qdrant.py
│   │   ├── search_dense.py
│   │   ├── search_keyword.py
│   │   ├── search_hybrid.py
│   │   └── search.py
│   │
│   └── evaluation/
│       ├── queries.json
│       ├── evaluate.py
│       └── inspect_chunks.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Embedding Dimension | 384 |
| Vector Database | Qdrant |
| Vector Similarity | Cosine similarity |
| Keyword Retrieval | BM25 |
| BM25 Library | `rank-bm25` |
| Dense Retrieval | BGE + Qdrant |
| Hybrid Retrieval | Weighted score fusion |
| Evaluation | Precision@3, Recall@3, MRR |

---

# 1. Corpus

The project uses technical AI/ML documents covering topics such as:

- HNSW
- Hybrid Search
- Semantic Search
- Retrieval-Augmented Generation
- Embedding Models
- BM25
- Vector Databases
- Chunking
- Reranking
- Vector Similarity

The corpus was expanded during the project to make the retrieval experiment more challenging.

---

# 2. Chunking

Long documents are divided into smaller text chunks before embedding.

The current corpus contains:

```text
10 documents
      ↓
70 chunks
```

Each chunk contains information such as:

```text
chunk_id
text
source
chunk_index
```

For example:

```text
hnsw_0
hnsw_1
hnsw_2
...
```

Chunking makes retrieval more useful because the system retrieves focused pieces of documents instead of entire long documents.

---

# 3. Embeddings

The project uses:

```text
BAAI/bge-small-en-v1.5
```

Each text chunk is converted into a dense vector with:

```text
384 dimensions
```

Conceptually:

```text
Text
  ↓
BGE Embedding Model
  ↓
[0.12, -0.43, 0.87, ...]
  ↓
384-dimensional vector
```

The same embedding model is used to convert a user's query into a vector.

This allows the system to compare the meaning of the query with the meaning represented by document chunks.

---

# 4. Vector Database

The project uses **Qdrant** to store and search the embeddings.

The current collection contains:

```text
70 points
384-dimensional vectors
COSINE distance
```

A stored Qdrant point conceptually contains:

```text
Point ID
    │
    ├── Vector
    │      └── 384 numbers
    │
    └── Payload
           ├── chunk_id
           ├── text
           ├── source
           └── chunk_index
```

The application-level `chunk_id` identifies the original chunk, while the Qdrant point ID identifies the stored vector point.

---

# 5. Dense Semantic Retrieval

Dense retrieval searches using vector representations.

The process is:

```text
User Query
    ↓
BGE Embedding Model
    ↓
Query Vector
    ↓
Qdrant
    ↓
Cosine Similarity
    ↓
Top-K Chunks
```

For example, a query such as:

```text
How does HNSW make vector search faster?
```

does not need to contain exactly the same wording as a relevant chunk.

Instead, the embedding model represents the query in semantic space and Qdrant retrieves vectors that are close to it.

The dense retrieval implementation is:

```text
src/retrieval/search_dense.py
```

---

# 6. Keyword Retrieval with BM25

BM25 is a lexical retrieval method.

Instead of representing text as dense embeddings, BM25 looks at the words that occur in the query and documents.

Conceptually:

```text
Query
  ↓
Token matching
  ↓
BM25 scoring
  ↓
Rank documents
  ↓
Top-K results
```

BM25 is particularly useful when the exact terminology matters.

For example, a query containing:

```text
HNSW layers
```

can strongly favor documents containing those terms.

The keyword retrieval implementation is:

```text
src/retrieval/search_keyword.py
```

---

# 7. Hybrid Retrieval

Hybrid retrieval combines:

- Dense semantic similarity
- BM25 keyword relevance

The project uses weighted score fusion.

Conceptually:

```text
Dense Score
     │
     ▼
Normalize
     │
     ├──────────────┐
     │              │
     │              ▼
     │        Weighted Sum
     │              ▲
     │              │
     └───────┐      │
             │      │
BM25 Score  │      │
     ↓      │      │
Normalize ──┘      │
                   ▼
             Hybrid Score
                   │
                   ▼
                Ranking
```

The formula is:

```text
Hybrid Score =
    Dense Weight × Normalized Dense Score
    +
    Keyword Weight × Normalized BM25 Score
```

Three weight configurations were evaluated:

| Configuration | Dense | Keyword |
|---|---:|---:|
| Hybrid 25/75 | 25% | 75% |
| Hybrid 50/50 | 50% | 50% |
| Hybrid 75/25 | 75% | 25% |

Normalization is necessary because dense similarity and BM25 scores are on different numerical scales.

The hybrid retrieval implementation is:

```text
src/retrieval/search_hybrid.py
```

---

# 8. Why Hybrid Search?

Keyword and semantic retrieval have different strengths.

### BM25

Good at:

- Exact terminology
- Technical keywords
- Named concepts
- Lexical matching

### Dense Retrieval

Good at:

- Semantic meaning
- Paraphrases
- Related concepts
- Queries where exact words differ

### Hybrid Retrieval

Attempts to combine both signals.

```text
                 Retrieval
                    │
          ┌─────────┴─────────┐
          │                   │
       BM25                Dense
          │                   │
   Exact wording        Semantic meaning
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
              Hybrid Ranking
```

---

# 9. Evaluation

The project uses a manually labeled evaluation set containing **10 queries**.

Example queries include:

```text
How does HNSW make vector search faster?

What is semantic search?

What is retrieval augmented generation?

How do embedding models convert text into vectors?

What is hybrid search?

What is a vector database?

What is reranking in information retrieval?

What is vector similarity?

What is cosine similarity used for in vector search?

How does HNSW navigate vectors to find similar results?
```

Each query has manually selected relevant chunk IDs.

This creates a ground-truth set against which retrieval results can be evaluated.

---

# 10. Evaluation Metrics

## Precision@3

Precision@3 measures how many of the top 3 retrieved results are relevant.

```text
Precision@3 =
Relevant results in top 3
─────────────────────────
           3
```

Example:

```text
Top 3:

✓ Relevant
✓ Relevant
✗ Not relevant

Precision@3 = 2/3 = 0.67
```

---

## Recall@3

Recall@3 measures how many of the known relevant chunks were retrieved in the top 3.

```text
Recall@3 =
Relevant results retrieved in top 3
───────────────────────────────────
       Total relevant chunks
```

Precision focuses on:

```text
How many retrieved results were relevant?
```

Recall focuses on:

```text
How many relevant results did we find?
```

---

## MRR

Mean Reciprocal Rank measures how early the first relevant result appears.

For one query:

```text
First relevant result at rank 1
→ Reciprocal Rank = 1
```

```text
First relevant result at rank 2
→ Reciprocal Rank = 1/2
```

```text
First relevant result at rank 3
→ Reciprocal Rank = 1/3
```

The reciprocal ranks are averaged across all queries.

A higher MRR means relevant results tend to appear earlier.

---

# 11. Experiment 1 — Initial Corpus

The first experiment used:

```text
5 documents
28 chunks
5 evaluation queries
```

The aggregate results were:

| Method | Precision@3 | Recall@3 | MRR |
|---|---:|---:|---:|
| Keyword / BM25 | 0.60 | 0.68 | 0.90 |
| Dense / BGE + Qdrant | 0.60 | 0.68 | 0.90 |
| Hybrid 25D/75K | 0.60 | 0.68 | 0.90 |
| Hybrid 50D/50K | 0.60 | 0.68 | 0.90 |
| Hybrid 75D/25K | 0.60 | 0.68 | 0.90 |

## Experiment 1 Interpretation

On the initial small corpus, all retrieval methods produced the same aggregate evaluation metrics.

However, identical aggregate metrics do not mean that every method produced identical rankings for every query.

The experiment therefore did not provide evidence that hybrid retrieval was universally better than BM25 or dense retrieval.

---

# 12. Experiment 2 — Expanded Corpus

The corpus was expanded from:

```text
5 documents
28 chunks
```

to:

```text
10 documents
70 chunks
```

The additional topics included:

- BM25
- Vector Databases
- Chunking
- Reranking
- Vector Similarity

The evaluation set was also expanded from **5 to 10 queries**, and the relevance labels were audited after the corpus expansion to account for newly relevant chunks.

---

# 13. Experiment 2 Results

The expanded experiment produced the following results:

| Method | Precision@3 | Recall@3 | MRR |
|---|---:|---:|---:|
| Keyword / BM25 | 0.4333 | 0.5250 | 0.7333 |
| Dense / BGE + Qdrant | 0.5333 | 0.6750 | 0.8500 |
| Hybrid 25D/75K | 0.5333 | 0.6833 | 0.8167 |
| Hybrid 50D/50K | 0.4667 | 0.6167 | 0.9500 |
| Hybrid 75D/25K | 0.5333 | 0.6750 | 0.9500 |

---

# 14. Experiment 2 Analysis

The expanded experiment showed clearer differences between retrieval strategies.

## Dense vs BM25

Dense retrieval achieved:

```text
Precision@3 = 0.5333
Recall@3    = 0.6750
MRR         = 0.8500
```

compared with BM25:

```text
Precision@3 = 0.4333
Recall@3    = 0.5250
MRR         = 0.7333
```

Therefore, on this evaluation set, dense retrieval outperformed BM25 across all three metrics.

---

## Hybrid Results

The **25% dense / 75% keyword** configuration achieved the highest Recall@3:

```text
Recall@3 = 0.6833
```

The **50% dense / 50% keyword** and **75% dense / 25% keyword** configurations achieved the highest MRR:

```text
MRR = 0.9500
```

The 75D/25K configuration matched dense retrieval on Precision and Recall while achieving a higher MRR.

No single hybrid configuration was best across every metric.

---

## Important Conclusion

The results suggest that combining dense and lexical signals can improve ranking behavior on some queries.

However, the evaluation contains only 10 manually labeled queries, so the results should be treated as directional experimental observations rather than general conclusions.

The experiment does **not** provide sufficient evidence to claim that hybrid retrieval universally outperforms BM25 or dense retrieval.

---

# 15. What Changed When the Corpus Expanded?

The corpus expansion increased the retrieval search space:

```text
28 chunks
    ↓
70 chunks
```

This introduced more candidate chunks that could compete for the top positions.

The expanded corpus also contained more overlapping AI/ML concepts.

For example, the query:

```text
What is semantic search?
```

could now retrieve chunks from:

```text
Semantic Search
BM25
Hybrid Search
Vector Database
```

This makes the retrieval task more challenging because several documents can contain related terminology without directly answering the query.

The evaluation labels were also updated and expanded after the corpus expansion, so metric changes should not be attributed solely to the larger corpus.

---

# 16. Important Vector Database Concepts

## Embeddings

Embeddings convert text into numerical vectors that capture semantic information.

```text
Text
 ↓
Embedding Model
 ↓
Dense Vector
```

---

## Vector Similarity

The system needs a way to determine how close two vectors are.

This project uses:

```text
Cosine similarity
```

Cosine similarity focuses primarily on the direction of vectors rather than their magnitude.

Conceptually:

```text
Query Vector
      \
       \   small angle
        \
         Document Vector
```

A smaller angle generally indicates greater similarity.

---

## HNSW

Qdrant uses approximate nearest-neighbor indexing techniques to make vector search efficient.

HNSW stands for:

```text
Hierarchical Navigable Small World
```

Instead of comparing a query against every vector exhaustively, graph-based search can navigate through promising regions of the vector space.

The important engineering trade-off is:

```text
Recall
  ↕
Latency
  ↕
Memory
```

Approximate nearest-neighbor search aims to achieve useful recall while reducing search cost.

This project uses Qdrant for vector retrieval but does not independently benchmark or tune HNSW parameters.

---

# 17. BM25 vs Dense vs Hybrid

| Property | BM25 | Dense | Hybrid |
|---|---|---|---|
| Exact keywords | Strong | Weaker | Strong |
| Semantic similarity | Limited | Strong | Strong |
| Paraphrases | Limited | Strong | Strong |
| Technical terminology | Strong | Strong | Strong |
| Implementation complexity | Low | Medium | Higher |
| Main signal | Token overlap | Vector similarity | Both |

No retrieval strategy is universally best.

The appropriate method depends on:

- Corpus
- Query type
- Relevance definition
- Embedding model
- Indexing method
- Ranking strategy
- Evaluation set

---

# 18. Running the Project

## 1. Create and activate the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Start Qdrant

Run Qdrant locally using Docker and make it available at:

```text
http://localhost:6333
```

## 4. Download the corpus

```bash
python src/ingestion/download.py
```

## 5. Chunk the documents

```bash
python src/ingestion/chunk.py
```

## 6. Generate embeddings

```bash
python src/embeddings/embed.py
```

## 7. Index the corpus into Qdrant

```bash
python src/retrieval/index_qdrant.py
```

## 8. Test dense retrieval

```bash
python src/retrieval/search_dense.py
```

## 9. Test keyword retrieval

```bash
python src/retrieval/search_keyword.py
```

## 10. Test hybrid retrieval

```bash
python src/retrieval/search_hybrid.py
```

## 11. Run the evaluation

```bash
python src/evaluation/evaluate.py
```

## 12. Run the interactive search engine

```bash
python src/retrieval/search.py
```

The interactive search engine allows a user to enter queries and compare the top results returned by BM25, dense retrieval, and hybrid retrieval.

---

# 19. Key Design Decisions

## Why BGE?

`BAAI/bge-small-en-v1.5` is a compact open-source embedding model suitable for demonstrating semantic retrieval without requiring a large hosted embedding API.

## Why Qdrant?

Qdrant provides vector storage, similarity search, metadata payloads, and approximate nearest-neighbor retrieval capabilities in a dedicated vector database.

## Why BM25?

BM25 provides a strong lexical baseline.

Without BM25, it would be difficult to determine whether semantic retrieval actually improves over traditional keyword retrieval.

## Why Hybrid Search?

Hybrid search allows the experiment to test whether combining lexical and semantic signals improves retrieval quality.

## Why manual relevance labels?

Retrieval metrics require a definition of which results are relevant.

For this project, relevance was manually judged for the evaluation queries.

---

# 20. Limitations

This project is intentionally a small retrieval experiment rather than a production-scale benchmark.

Current limitations include:

- Only 10 source documents
- 70 chunks
- Only 10 evaluation queries
- Manual relevance judgments
- Simple BM25 tokenization
- Simple min-max score normalization
- Simple weighted score fusion
- No automated test suite
- No large-scale latency benchmark
- No independent HNSW parameter tuning
- No cross-encoder reranking
- No Reciprocal Rank Fusion experiment

Because the evaluation set is small, the numerical results should be interpreted as experimental observations rather than general conclusions.

---

# 21. Future Improvements

Possible next experiments include:

- Expand the corpus substantially
- Add more evaluation queries
- Improve relevance judgments
- Experiment with chunk size and overlap
- Improve BM25 tokenization
- Compare additional embedding models
- Test metadata filtering
- Add cross-encoder reranking
- Compare weighted fusion with Reciprocal Rank Fusion
- Benchmark retrieval latency
- Tune HNSW parameters
- Visualize embedding space
- Measure performance at larger vector counts