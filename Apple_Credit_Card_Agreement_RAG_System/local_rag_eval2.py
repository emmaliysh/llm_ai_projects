"""
local_rag_eval.py
-----------------
Self-contained evaluation of a local RAG pipeline (FAISS + SentenceTransformer + Ollama/Gemma)
against the Apple Card Customer Agreement test suite.

Dependencies:
    pip install gradio pandas faiss-cpu sentence-transformers pypdf ollama python-dotenv

Usage:
    python local_rag_eval.py
"""

import json
import math
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import faiss
import ollama
import gradio as gr
import pandas as pd
from pypdf import PdfReader
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(override=True)

# ─────────────────────────────────────────────
# CONFIG  (change these to match your setup)
# ─────────────────────────────────────────────
PDF_FILE   = "Apple-Card-Customer-Agreement.pdf"   # path to the PDF
TESTS_FILE = "tests_claude.jsonl"                           # path to test cases
EMBED_MODEL = "all-MiniLM-L6-v2"                   # local embedding model
LLM_MODEL   = "gemma3:1b"                          # Ollama model for answering AND judging
JUDGE_MODEL = LLM_MODEL                            # same local model — no API key needed
CHUNK_SIZE  = 500                                  # words per chunk
CHUNK_OVERLAP = 50                                 # word overlap between chunks
TOP_K       = 3                                    # chunks to retrieve per query

# ─────────────────────────────────────────────
# DASHBOARD THRESHOLDS
# ─────────────────────────────────────────────
MRR_GREEN, MRR_AMBER           = 0.9, 0.75
NDCG_GREEN, NDCG_AMBER         = 0.9, 0.75
COVERAGE_GREEN, COVERAGE_AMBER = 90.0, 75.0
ANSWER_GREEN, ANSWER_AMBER     = 4.5, 4.0


# ═══════════════════════════════════════════════════════
# 1.  SCHEMAS
# ═══════════════════════════════════════════════════════

class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str


class RetrievalEval(BaseModel):
    mrr: float
    ndcg: float
    keywords_found: int
    total_keywords: int
    keyword_coverage: float


class AnswerEval(BaseModel):
    feedback: str = Field(description="Concise feedback on answer quality")
    accuracy: float = Field(description="Factual correctness vs reference answer (1–5)")
    completeness: float = Field(description="Coverage of reference answer content (1–5)")
    relevance: float = Field(description="How well the answer addresses the question (1–5)")


# ═══════════════════════════════════════════════════════
# 2.  RAG PIPELINE  (mirrors local_rag.py)
# ═══════════════════════════════════════════════════════

@dataclass
class RAGPipeline:
    """Encapsulates the local RAG pipeline so it is built once and reused."""
    pdf_path: str
    embed_model_name: str = EMBED_MODEL
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

    chunks: list[str]         = field(default_factory=list, init=False)
    index: object             = field(default=None,          init=False)
    embedder: object          = field(default=None,          init=False)

    def __post_init__(self):
        print("📄 Loading PDF …")
        reader = PdfReader(self.pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.chunks = self._chunk_text(text)
        print(f"   ✔ {len(self.chunks)} chunks created")

        print("🔢 Building embeddings …")
        self.embedder = SentenceTransformer(self.embed_model_name)
        vectors = self.embedder.encode(self.chunks).astype("float32")

        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        print("   ✔ FAISS index ready\n")

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i : i + self.chunk_size]))
        return chunks

    def fetch_context(self, question: str, k: int = TOP_K) -> list[str]:
        """Return the top-k relevant chunk strings."""
        q_vec = self.embedder.encode([question]).astype("float32")
        _, indices = self.index.search(q_vec, k)
        return [self.chunks[i] for i in indices[0]]

    def answer(self, question: str) -> tuple[str, list[str]]:
        """Return (answer, retrieved_chunks)."""
        context_chunks = self.fetch_context(question)
        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a legal assistant for Apple Card. "
            "Use the provided context to answer the user's question accurately and concisely.\n\n"
            f"CONTEXT:\n{context}"
        )
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
        )
        return response["message"]["content"], context_chunks


# ═══════════════════════════════════════════════════════
# 3.  TEST LOADER
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
# 4.  RETRIEVAL METRICS  (from eval.py)
# ═══════════════════════════════════════════════════════

def _reciprocal_rank(keyword: str, chunks: list[str]) -> float:
    kw = keyword.lower()
    for rank, chunk in enumerate(chunks, start=1):
        if kw in chunk.lower():
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def _ndcg(keyword: str, chunks: list[str], k: int = 10) -> float:
    kw = keyword.lower()
    rels = [1 if kw in c.lower() else 0 for c in chunks[:k]]
    dcg  = _dcg(rels, k)
    idcg = _dcg(sorted(rels, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, rag: RAGPipeline, k: int = 10) -> RetrievalEval:
    chunks = rag.fetch_context(test.question, k=k)

    mrr_scores  = [_reciprocal_rank(kw, chunks) for kw in test.keywords]
    ndcg_scores = [_ndcg(kw, chunks, k)         for kw in test.keywords]

    avg_mrr  = sum(mrr_scores)  / len(mrr_scores)  if mrr_scores  else 0.0
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    found    = sum(1 for s in mrr_scores if s > 0)
    total    = len(test.keywords)
    coverage = (found / total * 100) if total > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr, ndcg=avg_ndcg,
        keywords_found=found, total_keywords=total,
        keyword_coverage=coverage,
    )


# ═══════════════════════════════════════════════════════
# 5.  ANSWER EVALUATION  (LLM-as-judge, from eval.py)
# ═══════════════════════════════════════════════════════

def evaluate_answer(test: TestQuestion, rag: RAGPipeline) -> tuple[AnswerEval, str]:
    generated_answer, _ = rag.answer(test.question)

    # Ask the local Ollama model to judge, requesting strict JSON output
    judge_prompt = f"""You are an expert evaluator assessing RAG answer quality.
Compare the generated answer against the reference answer and score strictly.

Question: {test.question}

Generated Answer: {generated_answer}

Reference Answer: {test.reference_answer}

Score the generated answer on three dimensions:
1. accuracy (1-5): factual correctness vs reference. Use 1 if anything is wrong, 5 only if perfectly correct.
2. completeness (1-5): covers all key info from the reference. Use 5 only if ALL information is included.
3. relevance (1-5): directly answers the question with no off-topic content.

You MUST respond with ONLY a valid JSON object and nothing else. No explanation, no markdown, no code fences.
Use exactly this format:
{{"feedback": "brief feedback here", "accuracy": 3, "completeness": 3, "relevance": 3}}"""

    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        options={"temperature": 0},  # deterministic scoring
    )

    raw = response["message"]["content"].strip()

    # Strip markdown code fences if the model wrapped the JSON anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Extract first {...} block in case the model added extra text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        # Fallback: return neutral scores with the raw text as feedback
        return AnswerEval(feedback=raw[:300], accuracy=3, completeness=3, relevance=3), generated_answer

    try:
        data = json.loads(match.group())
        eval_result = AnswerEval(
            feedback=str(data.get("feedback", "No feedback")),
            accuracy=float(data.get("accuracy", 3)),
            completeness=float(data.get("completeness", 3)),
            relevance=float(data.get("relevance", 3)),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        eval_result = AnswerEval(feedback=raw[:300], accuracy=3, completeness=3, relevance=3)

    return eval_result, generated_answer


# ═══════════════════════════════════════════════════════
# 6.  GRADIO DASHBOARD
# ═══════════════════════════════════════════════════════

def _color(value: float, metric: str) -> str:
    if metric == "mrr":
        return "green" if value >= MRR_GREEN else ("orange" if value >= MRR_AMBER else "red")
    if metric == "ndcg":
        return "green" if value >= NDCG_GREEN else ("orange" if value >= NDCG_AMBER else "red")
    if metric == "coverage":
        return "green" if value >= COVERAGE_GREEN else ("orange" if value >= COVERAGE_AMBER else "red")
    if metric in ("accuracy", "completeness", "relevance"):
        return "green" if value >= ANSWER_GREEN else ("orange" if value >= ANSWER_AMBER else "red")
    return "black"


def _metric_card(label: str, value: float, metric: str,
                 pct: bool = False, score: bool = False) -> str:
    color = _color(value, metric)
    val_str = f"{value:.1f}%" if pct else (f"{value:.2f}/5" if score else f"{value:.4f}")
    return f"""
    <div style="margin:10px 0;padding:15px;background:#f5f5f5;border-radius:8px;
                border-left:5px solid {color};">
        <div style="font-size:13px;color:#666;margin-bottom:4px;">{label}</div>
        <div style="font-size:26px;font-weight:bold;color:{color};">{val_str}</div>
    </div>"""


def _complete_badge(count: int) -> str:
    return f"""
    <div style="margin-top:16px;padding:10px;background:#d4edda;border-radius:5px;
                text-align:center;border:1px solid #c3e6cb;">
        <span style="font-size:13px;color:#155724;font-weight:bold;">
            ✓ Evaluation complete — {count} tests
        </span>
    </div>"""


# ── lazy-init the pipeline once ─────────────────────────
_rag: RAGPipeline | None = None

def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline(pdf_path=PDF_FILE)
    return _rag


# ── Gradio callbacks ─────────────────────────────────────

def run_retrieval(progress=gr.Progress()):
    rag    = get_rag()
    tests  = load_tests()
    total  = len(tests)

    sum_mrr = sum_ndcg = sum_cov = 0.0
    cat_mrr: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []

    for i, test in enumerate(tests):
        r = evaluate_retrieval(test, rag)
        sum_mrr  += r.mrr
        sum_ndcg += r.ndcg
        sum_cov  += r.keyword_coverage
        cat_mrr[test.category].append(r.mrr)
        rows.append({
            "Question": test.question[:60] + "…",
            "Category": test.category,
            "MRR":      round(r.mrr, 4),
            "nDCG":     round(r.ndcg, 4),
            "Coverage %": round(r.keyword_coverage, 1),
        })
        progress((i + 1) / total, desc=f"Retrieval {i+1}/{total}")

    n = len(tests)
    avg_mrr  = sum_mrr  / n
    avg_ndcg = sum_ndcg / n
    avg_cov  = sum_cov  / n

    html = (
        _metric_card("Mean Reciprocal Rank (MRR)", avg_mrr,  "mrr")
        + _metric_card("Normalized DCG (nDCG)",    avg_ndcg, "ndcg")
        + _metric_card("Keyword Coverage",          avg_cov,  "coverage", pct=True)
        + _complete_badge(n)
    )

    cat_df = pd.DataFrame([
        {"Category": cat, "Average MRR": round(sum(v)/len(v), 4)}
        for cat, v in cat_mrr.items()
    ])
    detail_df = pd.DataFrame(rows)

    return html, cat_df, detail_df


def run_answers(progress=gr.Progress()):
    rag   = get_rag()
    tests = load_tests()
    total = len(tests)

    sum_acc = sum_comp = sum_rel = 0.0
    cat_acc: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []

    for i, test in enumerate(tests):
        ev, answer = evaluate_answer(test, rag)
        sum_acc  += ev.accuracy
        sum_comp += ev.completeness
        sum_rel  += ev.relevance
        cat_acc[test.category].append(ev.accuracy)
        rows.append({
            "Question":     test.question[:60] + "…",
            "Category":     test.category,
            "Accuracy":     ev.accuracy,
            "Completeness": ev.completeness,
            "Relevance":    ev.relevance,
            "Feedback":     ev.feedback,
        })
        progress((i + 1) / total, desc=f"Answer eval {i+1}/{total}")

    n = len(tests)
    avg_acc  = sum_acc  / n
    avg_comp = sum_comp / n
    avg_rel  = sum_rel  / n

    html = (
        _metric_card("Accuracy",     avg_acc,  "accuracy",     score=True)
        + _metric_card("Completeness", avg_comp, "completeness", score=True)
        + _metric_card("Relevance",    avg_rel,  "relevance",    score=True)
        + _complete_badge(n)
    )

    cat_df = pd.DataFrame([
        {"Category": cat, "Average Accuracy": round(sum(v)/len(v), 4)}
        for cat, v in cat_acc.items()
    ])
    detail_df = pd.DataFrame(rows)

    return html, cat_df, detail_df


# ═══════════════════════════════════════════════════════
# 7.  LAUNCH
# ═══════════════════════════════════════════════════════

def build_ui():
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Local RAG Evaluation Dashboard", theme=theme) as app:

        gr.Markdown("# 🏠 Local RAG Evaluation Dashboard")
        gr.Markdown(
            f"Evaluating **{LLM_MODEL}** (via Ollama) + **{EMBED_MODEL}** embeddings "
            f"on the Apple Card Customer Agreement test suite."
        )

        # ── RETRIEVAL ──────────────────────────────────────
        gr.Markdown("---\n## 🔍 Retrieval Evaluation")
        gr.Markdown(
            "Measures whether the FAISS retriever surfaces chunks "
            "that contain the expected keywords (MRR, nDCG, coverage)."
        )

        ret_btn = gr.Button("▶ Run Retrieval Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                ret_metrics = gr.HTML(
                    "<div style='padding:20px;text-align:center;color:#999;'>"
                    "Click the button above to start.</div>"
                )
            with gr.Column(scale=1):
                ret_chart = gr.BarPlot(
                    x="Category", y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1], height=350,
                )

        ret_table = gr.Dataframe(
            label="Per-question retrieval results",
            wrap=True,
        )

        ret_btn.click(
            fn=run_retrieval,
            outputs=[ret_metrics, ret_chart, ret_table],
        )

        # ── ANSWER QUALITY ─────────────────────────────────
        gr.Markdown("---\n## 💬 Answer Quality Evaluation")
        gr.Markdown(
            f"Uses **{JUDGE_MODEL}** as an LLM judge to score "
            "Accuracy, Completeness, and Relevance (1–5) against reference answers."
        )

        ans_btn = gr.Button("▶ Run Answer Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                ans_metrics = gr.HTML(
                    "<div style='padding:20px;text-align:center;color:#999;'>"
                    "Click the button above to start.</div>"
                )
            with gr.Column(scale=1):
                ans_chart = gr.BarPlot(
                    x="Category", y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5], height=350,
                )

        ans_table = gr.Dataframe(
            label="Per-question answer results (with LLM feedback)",
            wrap=True,
        )

        ans_btn.click(
            fn=run_answers,
            outputs=[ans_metrics, ans_chart, ans_table],
        )

    return app


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
