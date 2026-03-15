# RAG-Based Customer Service Chatbot for BOA Credit Cards

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Bank of America credit card agreements using grounded, document-backed responses — no hallucinations, no unsupported claims.

---

## 📌 Overview

Large financial institutions publish extensive credit card documentation that is difficult for customers to navigate. This project builds a conversational AI system that retrieves relevant passages from official credit card agreement PDFs and generates accurate, policy-faithful answers using GPT-o4-mini.

The system is designed strictly for **general policy Q&A** — it does not access personal account data or provide personalized financial advice.

---

## 🏗️ System Architecture

```
PDF Documents
     │
     ▼
LLM-Based PDF Parsing (→ Markdown)
     │
     ▼
Chunking (~800 chars with overlap)
     │
     ▼
Embedding (BAAI/bge-base-en-v1.5)
     │
     ▼
FAISS Vector Index
     │
     ▼
Query → Semantic Retrieval (Top 15 candidates)
     │
     ▼
Cross-Encoder Reranking (Top 6 passages)
     │
     ▼
Answer Generation (GPT-o4-mini)
```

---

## ✨ Features

- **LLM-based PDF parsing** — converts complex PDFs (tables, multi-column layouts) into clean Markdown before indexing
- **Semantic retrieval** — FAISS vector search with BAAI/bge-base-en-v1.5 sentence embeddings
- **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 refines candidate passage ranking for higher precision
- **Query expansion** — bridges informal user language to formal agreement terminology (e.g., "ATM withdrawal" → "Bank Cash Advance")
- **Prompt engineering** — GPT-5-mini is constrained via prompt instructions to answer only from retrieved context
- **Policy-gating** — filters out account-specific or advisory queries outside the system's scope
- **LLM-as-judge evaluation** — automated scoring of accuracy, completeness, and relevance

---

## 📂 Dataset

- **23** publicly available Bank of America credit card agreement PDFs from https://www.bankofamerica.com/credit-cards/credit-card-agreements/
- **200+** pages of financial and legal content
- Chunked into ~800-character segments with overlap to preserve cross-sentence policy clauses
- **40** LLM-generated evaluation test cases, manually reviewed for correctness

---

## 📊 Results

### Retrieval Performance

| Metric    | Value  |
|-----------|--------|
| Precision | 0.85   |
| Recall    | 0.94   |
| MRR       | 0.8417 |
| nDCG      | 0.8463 |

### Answer Quality (LLM-as-Judge, 1–5 scale)

| Metric       | RAG System | LLM-Only Baseline |
|--------------|------------|-------------------|
| Accuracy     | 4.4        | 2.3 (+42%)        |
| Completeness | 4.0        | 2.1               |
| Relevance    | 4.8        | 4.2               |

---

## 🛠️ Tech Stack

| Component         | Tool / Model                        |
|-------------------|-------------------------------------|
| PDF Parsing       | LLM-based → Markdown                |
| Embedding Model   | BAAI/bge-base-en-v1.5               |
| Vector Index      | FAISS (IndexFlatL2)                 |
| Reranker          | ms-marco-MiniLM-L-6-v2             |
| Language Model    | GPT-o4-mini (OpenAI API)             |
| Evaluation        | LLM-as-judge framework              |
| Language          | Python                              |

---





## 📄 License

This project is for academic purposes only. All credit card agreement documents used are publicly available.
