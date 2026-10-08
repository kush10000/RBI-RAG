"""M3: retrieve -> (maybe refuse) -> Claude answers from sources with citations.

Two-layer "I don't know":
  Layer 1 (cheap, no LLM call): if even the BEST chunk scores below MIN_SCORE, nothing in the
           corpus is about this topic -> refuse immediately. Catches off-topic ("capital of France").
  Layer 2 (LLM): the prompt forbids outside knowledge and tells Claude to start with
           "I don't know" when the sources don't contain the answer. Catches on-topic but
           unanswerable questions (the NRI one), which similarity alone cannot separate.
"""
import numpy as np
import psycopg
from anthropic import Anthropic
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from embeddings import embed_query
from index import DSN

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"  # cheap + fast; swap for a bigger model and re-run eval to compare
TOP_K = 5            # distinct chunks sent to Claude
MIN_SCORE = 0.55     # layer 1 cutoff: between off-topic (~0.42) and answerable (~0.70). Tuned by eval.py.
DUP_SIM = 0.97       # two retrieved chunks this similar = same text (e.g. Payments Banks vs Local Area Banks)

SYSTEM_PROMPT = """You answer questions about Reserve Bank of India (RBI) circulars.

Rules:
1. Use ONLY the numbered sources provided. Never use outside knowledge, even if you know the answer.
2. Cite every factual claim with its source number in square brackets, e.g. [2].
3. If the sources answer the question fully or partly, answer with what they say. If something is missing, mention the gap at the END. Begin your reply with exactly "I don't know" ONLY when the sources contain nothing that answers the question; then say in one sentence what the sources do cover.
4. If a source says its text also appears in other documents, it applies equally to all of them.
5. Be concise: 1-4 sentences."""

_client = None


def _claude() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _pages(start: int, end: int) -> str:
    """Citation label: "4" or "4-5" when a chunk crosses a page break."""
    return str(start) if start == end else f"{start}-{end}"


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Top-k DISTINCT chunks. Near-duplicates are merged and remembered in `also_in`."""
    qvec = embed_query(question)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT doc, page, page_end, text, embedding, 1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, qvec, k * 3),  # over-fetch so we still have k left after removing duplicates
        ).fetchall()

    kept: list[dict] = []
    for doc, page, page_end, text, emb, score in rows:
        # pgvector returns its own Vector type, not a numpy array -> convert explicitly.
        emb = (emb.to_numpy() if hasattr(emb, "to_numpy") else np.asarray(emb)).astype(np.float32)
        # Your idea: if a chunk's vector is almost identical to one we already kept, it says the
        # same thing. Vectors are normalized, so the dot product IS the cosine similarity.
        dup = next((c for c in kept if float(c["emb"] @ emb) >= DUP_SIM), None)
        if dup:
            dup["also_in"].append((doc, _pages(page, page_end)))
            continue
        kept.append({"doc": doc, "page": _pages(page, page_end), "text": text, "score": float(score),
                     "emb": emb, "also_in": []})
        if len(kept) == k:
            break
    return kept


def _format_source(i: int, c: dict) -> str:
    """One numbered source for the prompt. Claude only knows what is written here, so the
    documents merged away by dedupe MUST be listed, or Claude thinks the text belongs to one doc only."""
    header = f"[{i}] {c['doc']} (pages {c['page']})"
    if c["also_in"]:
        others = "; ".join(f"{d} (pages {p})" for d, p in c["also_in"])
        header += f"\n    Identical text also appears in: {others}"
    return f"{header}\n{c['text']}"


def _is_refusal(text: str) -> bool:
    return text.lower().replace("’", "'").startswith("i don't know")


def answer(question: str) -> dict:
    chunks = retrieve(question)
    top_score = chunks[0]["score"] if chunks else 0.0

    # Layer 1: nothing relevant at all -> don't spend an LLM call.
    if top_score < MIN_SCORE:
        return {"answer": "I don't know. Nothing in the indexed RBI documents is relevant to this question.",
                "refused": True, "refused_by": "threshold", "top_score": top_score, "chunks": chunks}

    sources = "\n\n".join(_format_source(i, c) for i, c in enumerate(chunks, start=1))
    resp = _claude().messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"}],
    )
    text = resp.content[0].text.strip()

    # Layer 2: Claude read the sources and said they don't answer it.
    refused = _is_refusal(text)
    return {"answer": text, "refused": refused, "refused_by": "llm" if refused else None,
            "top_score": top_score, "chunks": chunks,
            "tokens": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}}
