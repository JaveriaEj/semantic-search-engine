# Semantic Search Engine

A production-style semantic retrieval system that compares **keyword search, dense semantic search, and hybrid search** over a real AI/ML technical corpus.

The project demonstrates the complete retrieval pipeline:

**Documents → Chunking → Embeddings → Qdrant → Dense Search / BM25 → Hybrid Search → Evaluation**

The goal is not only to build a working search engine, but to understand and measure the trade-offs between lexical and semantic retrieval.

---

# 1. Project Overview

Traditional keyword search primarily looks for matching words.

Semantic search looks for matching meaning.

For example:

> How does HNSW make vector search faster?

A keyword system focuses on terms such as:

- HNSW
- vector
- search
- faster

A semantic retrieval system converts the query into a vector and searches for document chunks with similar semantic meaning.

This project implements:

1. Real technical document ingestion
2. Text chunking
3. Dense embeddings
4. Qdrant vector storage
5. Dense semantic retrieval
6. BM25 keyword retrieval
7. Hybrid retrieval
8. Retrieval evaluation
9. Precision@3
10. Recall@3
11. Mean Reciprocal Rank (MRR)

---

# 2. Project Goal

The project was designed to answer a practical AI engineering question:

> **Does combining semantic and keyword retrieval improve retrieval quality compared with using either method alone?**

Three retrieval approaches are compared:

```text
Keyword Search
      |
      |----> BM25
      |
      v
Lexical ranking


Dense Search
      |
      |----> Embeddings
      |----> Qdrant
      |
      v
Semantic ranking


Hybrid Search
      |
      |----> Dense score
      |----> BM25 score
      |
      v
Combined ranking