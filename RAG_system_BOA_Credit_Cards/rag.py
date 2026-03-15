"""
boa_answers.py
--------------
Streamlined RAG pipeline — answers only, no LLM judge.
Outputs a clean JSON file with questions + generated answers
ready to paste into your external evaluator.

Also prints a retrieval evaluation table per question:
  - Precision: fraction of retrieved chunks that contain ANY keyword
                (chunk-level, not keyword-level)
  - Recall:    fraction of retrieved chunks that contain ALL keywords
                found anywhere / total keywords expected
  - MRR:       mean reciprocal rank across keywords
  - nDCG:      normalised discounted cumulative gain across keywords

Dependencies:
    pip install pandas faiss-cpu sentence-transformers openai python-dotenv

Usage:
    python boa_answers.py
"""

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import faiss
import openai
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv(override=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MD_FOLDER      = "md_claude_no_first_page"   # folder containing .md source files
TESTS_FILE     = "10test.jsonl"              # path to test cases
EMBED_MODEL    = "BAAI/bge-base-en-v1.5"    # local embedding model
LLM_MODEL      = "gpt-4o-mini"              # OpenAI model for answering
CHUNK_SIZE     = 800                         # chars per chunk (increased from 500)
CHUNK_OVERLAP  = 150                         # char overlap between chunks
TOP_K          = 15                          # chunks to retrieve per query
RERANK_TOP_K   = 6                          # top chunks kept after reranking (increased from 3)
OUTPUT_FILE    = "rag_answers.json"          # where to save results

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cross_encoder  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ═══════════════════════════════════════════════════════
# 1.  SCHEMAS
# ═══════════════════════════════════════════════════════

class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str


# ═══════════════════════════════════════════════════════
# 2.  RETRIEVAL METRICS
# ═══════════════════════════════════════════════════════

def _reciprocal_rank(keyword: str, chunks: list[str]) -> float:
    """Rank of first chunk containing the keyword."""
    kw = keyword.lower()
    for rank, chunk in enumerate(chunks, start=1):
        if kw in chunk.lower():
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def _ndcg(keyword: str, chunks: list[str]) -> float:
    kw   = keyword.lower()
    rels = [1 if kw in c.lower() else 0 for c in chunks]
    dcg  = _dcg(rels, len(rels))
    idcg = _dcg(sorted(rels, reverse=True), len(rels))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(keywords: list[str], chunks: list[str]) -> dict:
    """
    Chunk-based precision and recall:
      Precision = fraction of retrieved chunks containing at least one keyword
      Recall    = fraction of keywords found in at least one retrieved chunk
    MRR and nDCG are keyword-averaged.
    """
    n_chunks = len(chunks)

    # Chunk-level precision: chunk is relevant if it contains ANY keyword
    relevant_chunks = sum(
        1 for chunk in chunks
        if any(kw.lower() in chunk.lower() for kw in keywords)
    )
    precision = relevant_chunks / n_chunks if n_chunks > 0 else 0.0

    # Keyword-level recall: keyword is found if it appears in ANY chunk
    found_keywords = sum(
        1 for kw in keywords
        if any(kw.lower() in chunk.lower() for chunk in chunks)
    )
    recall = found_keywords / len(keywords) if keywords else 0.0

    # MRR and nDCG averaged across keywords
    mrr_scores  = [_reciprocal_rank(kw, chunks) for kw in keywords]
    ndcg_scores = [_ndcg(kw, chunks)            for kw in keywords]
    avg_mrr     = sum(mrr_scores)  / len(mrr_scores)  if mrr_scores  else 0.0
    avg_ndcg    = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return {
        "precision":       precision,
        "recall":          recall,
        "mrr":             avg_mrr,
        "ndcg":            avg_ndcg,
        "relevant_chunks": relevant_chunks,
        "total_chunks":    n_chunks,
        "found_keywords":  found_keywords,
        "total_keywords":  len(keywords),
    }


# ═══════════════════════════════════════════════════════
# 3.  RAG PIPELINE
# ═══════════════════════════════════════════════════════

@dataclass
class RAGPipeline:
    md_folder: str
    embed_model_name: str = EMBED_MODEL
    chunk_size: int       = CHUNK_SIZE
    chunk_overlap: int    = CHUNK_OVERLAP

    chunks:   list[str] = field(default_factory=list, init=False)
    index:    object    = field(default=None,          init=False)
    embedder: object    = field(default=None,          init=False)

    def __post_init__(self):
        print("📄 Loading Markdown files …")
        md_files = list(Path(self.md_folder).glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No .md files found in '{self.md_folder}'")
        print(f"   Found {len(md_files)} file(s)")

        all_text = []
        for md_file in md_files:
            print(f"   Reading {md_file.name} …")
            all_text.append(md_file.read_text(encoding="utf-8"))

        self.chunks = self._chunk_text("\n\n".join(all_text))
        print(f"   ✔ {len(self.chunks)} chunks created\n")

        print("🔢 Building FAISS index …")
        self.embedder = SentenceTransformer(self.embed_model_name)
        vectors = self.embedder.encode(self.chunks).astype("float32")
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        print("   ✔ FAISS index ready\n")

    def _chunk_text(self, text: str) -> list[str]:
        """
        Paragraph-aware chunker with overlap.
        Splits on blank lines; never cuts mid-paragraph.
        """
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # carry last paragraph as overlap
                overlap_text  = current_chunk[-1]
                current_chunk = [overlap_text]
                current_size  = len(overlap_text)

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    # Old single-query fetch (kept for reference):
    # def fetch_context(self, question: str, k: int = TOP_K) -> list[str]:
    #     q_vec = self.embedder.encode([question]).astype("float32")
    #     _, indices = self.index.search(q_vec, k)
    #     return [self.chunks[i] for i in indices[0]]

    def fetch_context(self, question: str, k: int = TOP_K) -> list[str]:
        queries = [question]

        # Only add expanded query if it actually differs from the original
        expanded = (
            question
            .replace("pay in full", "Grace Period Balance Paid in Full")
            .replace("ATM withdrawals", "Bank Cash Advance ATM")
            .replace("late fee", "Late Fee billing cycles")
        )
        if expanded != question:
            queries.append(expanded)
            print(f"   [query expansion] added: {expanded[:80]}…")

        seen, results = set(), []
        for q in queries:
            q_vec = self.embedder.encode([q]).astype("float32")
            _, indices = self.index.search(q_vec, k)
            for i in indices[0]:
                if i not in seen:
                    seen.add(i)
                    results.append(self.chunks[i])
        return results

    def rerank(self, question: str, chunks: list[str]) -> list[str]:
        scores = cross_encoder.predict([(question, chunk) for chunk in chunks])
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in top_indices[:RERANK_TOP_K]]

    def answer(self, question: str) -> tuple[str, list[str]]:
        # """Retrieve, rerank, and generate an answer. Returns (answer_text, chunks)."""
        # context_chunks = self.fetch_context(question)
        # context_chunks = self.rerank(question, context_chunks)

        # context = "\n\n---\n\n".join(context_chunks)  # full chunk, no truncation

        context_chunks = self.fetch_context(question)
        context_chunks = self.rerank(question, context_chunks)

        # ── DEBUG ──────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"CHUNKS FOR: {question[:60]}")
        print(f"{'='*60}")
        for idx, chunk in enumerate(context_chunks, 1):
            print(f"\n--- Chunk {idx} ---")
            print(chunk)
        print(f"{'='*60}\n")
        # ── END DEBUG ──────────────────────────────────────

        context = "\n\n---\n\n".join(context_chunks)

        # system_prompt = (
        #     "You are a Credit Card Agreement Specialist performing a compliance audit. "
        #     "Your job is to extract policy details with zero omissions.\n\n"

        #     "STEP 1 — SCAN: Read the text and identify every sentence relevant to the question.\n"
        #     "STEP 2 — EXTRACT: From those sentences, collect:\n"
        #     "   • Every percentage (e.g. 19.49%, 27.49%)\n"
        #     "   • Every margin or index component (e.g. Prime Rate + 12.74%)\n"
        #     "   • Every dollar amount (e.g. $89, $35, $40)\n"
        #     "   • Every date or reference period (e.g. as of 12/31/2025)\n"
        #     "   • Every condition, exception, or qualifying clause (e.g. 'if', 'unless', 'provided that')\n"
        #     "   • Every sequential rule (e.g. 'first...then...')\n"
        #     "STEP 3 — VERIFY: Before writing your answer, ask yourself: "
        #     "'Is there any number, condition, or exception in the relevant sentences that I have not yet included?' "
        #     "If yes, add it.\n"
        #     "STEP 4 — WRITE: Write 1–4 complete sentences using the exact terminology from the text. "
        #     "Do not paraphrase numbers. Do not drop secondary figures like margins or reference dates. "
        #     "Do not hedge with words like 'typically' or 'generally'. "
        #     "Do not add outside knowledge. Do not use bullet points.\n\n"

        #     "If the answer is not in the text, respond exactly: 'Not found in agreement.'\n\n"

        #     f"TEXT:\n{context}\n\n"
        #     f"QUESTION: {question}\n\n"
        #     "AUDIT ANSWER:"
        # )

        system_prompt = (
            "You are a Credit Card Agreement Specialist performing a compliance audit. "
            "Your job is to extract policy details with zero omissions.\n\n"

            "IMPORTANT — TERMINOLOGY BRIDGING:\n"
            "The question may use everyday language while the agreement uses formal terms. "
            "You MUST map these before searching:\n"
            "   • 'ATM withdrawals' or 'ATM' → look for 'Bank Cash Advance' or 'ATM Cash Advance'\n"
            "   • 'pay in full' → look for 'Grace Period Balance', 'Paid in Full', 'Interest Saving Balance'\n"
            "   • 'penalty rate' → look for 'Penalty APR'\n"
            "   • 'late payment fee' → look for 'Late Fee'\n"
            "Never say 'Not found' because the wording differs — find the formal equivalent and answer from it.\n\n"

            "STEP 1 — SCAN: Identify every sentence or bullet in the text relevant to the question, "
            "including using the formal equivalents above.\n"
            "STEP 2 — EXTRACT: Collect every percentage, margin, dollar amount, date, "
            "condition, exception, and sequential rule from those sentences.\n"
            "STEP 3 — VERIFY: Ask yourself: have I included every number, condition, "
            "and exception? If not, add it.\n"
            "STEP 4 — WRITE: 1–4 complete sentences using exact terminology from the text. "
            "Do not hedge. Do not add outside knowledge. Do not use bullet points.\n\n"

            "Only respond 'Not found in agreement.' if after thorough search using ALL formal "
            "equivalents, the topic is genuinely absent from the text.\n\n"

            f"TEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "AUDIT ANSWER:"
        )

        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip(), context_chunks


# ═══════════════════════════════════════════════════════
# 4.  TEST LOADER
# ═══════════════════════════════════════════════════════

def load_tests(path: str = TESTS_FILE) -> list[TestQuestion]:
    tests = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(TestQuestion(**json.loads(line)))
    return tests


# ═══════════════════════════════════════════════════════
# 5.  MAIN — run all questions and save answers
# ═══════════════════════════════════════════════════════

def main():
    rag   = RAGPipeline(md_folder=MD_FOLDER)
    tests = load_tests()
    total = len(tests)

    results     = []
    ret_results = []

    for i, test in enumerate(tests, start=1):
        print(f"[{i}/{total}] {test.question[:70]}…")
        generated, chunks = rag.answer(test.question)
        ret = evaluate_retrieval(test.keywords, chunks)
        print(f"        → {generated[:100]}…")
        print(f"           retrieval: precision={ret['precision']:.2f}  recall={ret['recall']:.2f}  "
              f"mrr={ret['mrr']:.4f}  ndcg={ret['ndcg']:.4f}\n")

        results.append({
            "id":               i,
            "category":         test.category,
            "question":         test.question,
            "generated_answer": generated,
            "reference_answer": test.reference_answer,
            "keywords":         test.keywords,
            "retrieval":        ret,
        })
        ret_results.append(ret)

    # ── Save JSON ────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Answers saved to '{OUTPUT_FILE}'\n")

    # ── Answer summary table ─────────────────────────────
    print("=" * 130)
    print("ANSWER SUMMARY")
    print("=" * 130)
    print(f"{'#':<4} {'Cat':<12} {'Question':<52} {'Answer (truncated)'}")
    print("-" * 130)
    for r in results:
        print(f"{r['id']:<4} {r['category']:<12} {r['question'][:50]:<52} {r['generated_answer'][:60]}…")

    # ── Retrieval evaluation table ───────────────────────
    print("\n" + "=" * 90)
    print("RETRIEVAL EVALUATION  (chunk-based precision & recall)")
    print("=" * 90)
    print(f"  Precision = relevant chunks / total retrieved chunks")
    print(f"  Recall    = keywords found in chunks / total keywords")
    print(f"  MRR/nDCG  = keyword-averaged rank quality\n")
    print(f"{'#':<4} {'Cat':<12} {'Prec':>6} {'Recall':>7} {'MRR':>7} {'nDCG':>7} "
          f"{'RelChk':>7} {'TotChk':>7} {'KwFound':>8} {'KwTotal':>8}")
    print("-" * 90)

    sum_prec = sum_rec = sum_mrr = sum_ndcg = 0.0
    for i, (r, ret) in enumerate(zip(results, ret_results), start=1):
        print(f"{i:<4} {r['category']:<12} "
              f"{ret['precision']:>6.2f} {ret['recall']:>7.2f} "
              f"{ret['mrr']:>7.4f} {ret['ndcg']:>7.4f} "
              f"{ret['relevant_chunks']:>7} {ret['total_chunks']:>7} "
              f"{ret['found_keywords']:>8} {ret['total_keywords']:>8}")
        sum_prec += ret["precision"]
        sum_rec  += ret["recall"]
        sum_mrr  += ret["mrr"]
        sum_ndcg += ret["ndcg"]

    n = len(ret_results)
    print("-" * 90)
    print(f"{'AVG':<4} {'':<12} "
          f"{sum_prec/n:>6.2f} {sum_rec/n:>7.2f} "
          f"{sum_mrr/n:>7.4f} {sum_ndcg/n:>7.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()