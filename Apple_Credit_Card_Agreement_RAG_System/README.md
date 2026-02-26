# Apple Card Agreement RAG System with Evaluation Pipeline

A fully local Retrieval-Augmented Generation (RAG) system that indexes the Apple Card Customer Agreement PDF and answers questions about it — with a built-in evaluation dashboard to measure retrieval and answer quality. No API keys required.


## Tech Stack

| Component | Tool |
|---|---|
| LLM | [Ollama](https://ollama.com/) + `gemma3:1b` |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Store | FAISS (CPU) |
| PDF Parsing | PyPDF |
| Dashboard | Gradio |
| Test Cases | Claude-generated JSONL |


## Features

- **Local RAG pipeline** — chunks the PDF, embeds with Sentence Transformers, and stores vectors in FAISS for fast similarity search
- **Retrieval evaluation** — scores each test question on MRR, nDCG, and keyword coverage
- **LLM-as-judge answer evaluation** — uses Gemma locally to score generated answers on Accuracy, Completeness, and Relevance (1–5) against reference answers
- **Gradio dashboard** — interactive UI with per-question breakdowns, bar charts by category, and color-coded metric cards


## Setup

### 1. Install dependencies

```bash
pip install -r requirements2.txt
```

### 2. Install and start Ollama

```bash
# Install Ollama: https://ollama.com/download
ollama pull gemma3:1b
```

### 3. Add the source PDF

Download the [Apple Card Customer Agreement](https://www.apple.com/legal/apple-card/customer-agreement/) and place it in the project root as:

```
Apple-Card-Customer-Agreement.pdf
```

### 4. Run the dashboard

```bash
python local_rag_eval2.py
```

The Gradio dashboard will open in your browser automatically.

---

## Test Case Format

Test cases are stored in `tests_claude.jsonl`, one JSON object per line:

```json
{
  "question": "What is the maximum APR Apple Card can charge?",
  "keywords": ["APR", "maximum", "variable"],
  "reference_answer": "The maximum APR is 29.99%.",
  "category": "direct_fact"
}
```

Categories:
- `direct_fact` — answer is contained in a single chunk
- `spanning` — answer requires combining information across multiple chunks

---

## Configuration

Edit the constants at the top of `local_rag_eval2.py` to customize:

```python
PDF_FILE      = "Apple-Card-Customer-Agreement.pdf"
TESTS_FILE    = "tests_claude.jsonl"
EMBED_MODEL   = "all-MiniLM-L6-v2"
LLM_MODEL     = "gemma3:1b"
CHUNK_SIZE    = 500    # words per chunk
CHUNK_OVERLAP = 50     # overlap between chunks
TOP_K         = 3      # chunks retrieved per query
```

---

## Notes

- Swap `LLM_MODEL` to any model available in your local Ollama installation (e.g. `llama3`, `mistral`) to benchmark different models
- The same model is used for both answering and judging — no external judge API needed
- FAISS index is rebuilt on every run (in-memory); add persistence if needed for large documents
