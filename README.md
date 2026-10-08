# RBI-RAG: question answering over RBI circulars, built from scratch

Ask questions about Reserve Bank of India circulars and get answers with page-level citations.
If the documents don't cover a question, the system says "I don't know" instead of guessing.

No LangChain or LlamaIndex. Every stage (extraction, chunking, embedding, vector search, prompting,
refusal, evaluation) is hand-written, so each decision below is one I made and measured.

**Results on a 21-question eval set:** 100% retrieval recall@5 · **94% answer accuracy** · **100% correct refusals** on unanswerable and trap questions.

---

## Demo

```text
> python ask.py "Who can get one-time approval to buy more shares in a payments bank?"
A: Qualifying persons ... mutual funds registered with SEBI, pension funds registered with PFRDA,
   or insurance companies registered with IRDAI, provided they do not belong to the promoter group [3].
   The one-time approval allows acquisitions up to 10 per cent of paid-up share capital [1].
Sources: [1] Local Area Banks ... (identical text also in: Payments Banks ...) ...

> python ask.py "Can an NRI keep money in a rupee account in India?"
A: I don't know. The sources cover NRD-CSR reporting procedures for banks, but do not address
   whether NRIs may maintain rupee accounts.                        (refused by: LLM, best score 0.656)

> python ask.py "What is the capital of France?"
A: I don't know. Nothing in the indexed RBI documents is relevant.  (refused by: threshold, 0.436, no LLM call)
```

## Architecture

```text
INDEXING (python ingest.py && python index.py)
  9 RBI PDFs ─► pypdf text ─► clean (Hindi, ▯ glyphs, ____ / .... leaders, page numbers)
             ─► whole-document chunks (1200 chars, 200 overlap, page range kept)
             ─► bge-small-en-v1.5 embeddings (local, 384-d) ─► Postgres + pgvector

QUERYING (python ask.py "...")
  question ─► embed (same model + query prefix) ─► top-15 by cosine distance (<=>)
           ─► dedupe near-identical chunks (cos ≥ 0.97), keep top 5, remember merged docs
           ─► Layer 1: best score < 0.55 ?  → "I don't know" (no LLM call)
           ─► Layer 2: Claude Haiku 4.5, sources only, cite [n], "I don't know" if nothing answers
```

## Key decisions and tradeoffs

| Decision | Why | Tradeoff |
|---|---|---|
| **Local embeddings** (bge-small-en-v1.5) instead of an API | RBI-regulated clients often can't send documents to third-party APIs; no per-token cost or rate limits | Smaller model than hosted options; English-only, so Hindi text is stripped |
| **Postgres + pgvector** instead of a dedicated vector DB | One system to run; vectors, metadata and (later) full-text search in the same SQL | Less specialised tuning than Pinecone and similar |
| **No ANN index (HNSW)** | 54 chunks: an exact scan takes milliseconds and is 100% accurate | Add HNSW past ~100k chunks |
| **Chunks of 1200 chars with 200 overlap** | bge-small silently truncates after 512 tokens; ~1200 chars ≈ 300 tokens fits safely | Smaller chunks = sharper retrieval but less context per chunk |
| **One `embeddings.py`** for both documents and queries | Embeddings from different models live in unrelated spaces; same dimensions = silently wrong results | None; this is a guard rail |
| **Two-layer "I don't know"** | Similarity measures *topic*, not *whether the answer is there* (see below) | One extra LLM call for on-topic questions |
| **Doc-level recall in eval** | Survives re-chunking, so chunking strategies can be compared | Can hide "right document, wrong chunk" (it did; see v3) |
| **Keyword-match answer scoring** | Deterministic, transparent, free | Cruder than LLM-as-judge; needs several accepted phrasings per answer |

### Why one similarity threshold can't detect "not in the documents"

| Question type | Best-chunk score |
|---|---|
| Answerable | 0.674 – 0.814 |
| On-topic but **not answerable** (e.g. "max soiling for a ₹2000 note", where the table only covers ₹5–₹500) | up to **0.764** |
| Off-topic ("capital of France") | 0.414 |

The ₹2000 trap scores higher than several answerable questions. A threshold (0.55) cheaply removes
off-topic questions; only the LLM, reading the actual text, can tell "on-topic" from "answered".
Unrelated text still scores ~0.4 rather than 0 because sentence embeddings occupy a narrow cone of
directions (anisotropy), so thresholds must be tuned per model and corpus.

## Evaluation

`eval_set.json`: 21 questions. 16 answerable, spread across 7 documents, and 5 that must be refused:
2 on-topic traps, 2 RBI-related but out of corpus, and 1 off-topic.

| Version | Change | Recall@5 | Answer accuracy | False refusals | Correct refusals |
|---|---|---|---|---|---|
| v1 | Baseline (page-by-page chunks, dedupe, strict refusal prompt) | 100% | 13/16 (81%) | 3 | 5/5 |
| v2 | Pass merged-duplicate document names into the prompt; refuse only when *nothing* answers | 100% | 13/16 (81%) | 3 | 5/5 |
| v3 | Chunk the whole document across page breaks; cite page ranges | 100% | **15/16 (94%)** | **1** | **5/5** |

v2 fixed a real bug that the demo exposed, but the eval didn't move: the three eval failures had a
different root cause (chunk boundaries), which v3 fixed. Without the eval I would have shipped v2
believing the false refusals were solved.

Cost: ~1.7k input / ~100 output tokens per answered question on Haiku 4.5.

## What broke (and what it taught me)

1. **Extraction junk.** RBI PDFs are bilingual, and pypdf output contained Hindi, `▯` replacement glyphs for undecodable fonts, `_____` form blanks and `......` table-of-contents leaders. A letterhead full of `▯` ranked #1 for an NRI question. Fix: regex cleaning. *Look at your chunks before you embed them.*
2. **Duplicate documents.** The Payments Banks and Local Area Banks circulars are near word-for-word identical (the same amendment applied to two Master Directions), so duplicates filled the top-5. Fix: merge chunks with cosine ≥ 0.97.
3. **The dedupe fix caused a new bug.** Merging kept the "Local Area Banks" label, and the prompt never said the text also belonged to Payments Banks. Claude, correctly following "use only the sources", refused a payments-bank question. *The LLM only knows what is in the prompt; metadata dropped in the pipeline is information lost.*
4. **Over-strict refusal rule.** "Begin with I don't know if the sources don't contain the answer" made Claude hedge correct partial answers ("I don't know the complete answer… [correct answer]"), which counted as refusals. Fix: refuse only when nothing answers, and state gaps at the end. Refusal accuracy stayed at 5/5, so the loosening didn't let trap questions through.
5. **Page-by-page chunking split answers.** A known day-1 tradeoff: chunking per page gave exact citations but split sentences across page breaks ("…five per cent of the | *page 5* | banking company … within three working days"). Doc-level recall was 100%, so the retrieval metric hid it. Fix: chunk the whole document and track page ranges with character offsets.
6. **Library boundary types.** Writing numpy vectors into pgvector worked, but reading them back returned a `Vector` object and crashed numpy. Fix: `.to_numpy()`.

## Known limitations and next steps

- **pb-3 still fails** ("within how many working days must a qualifying person report its holding crossing five percent?"). PB_3_ANALYSIS
- **Letterhead boilerplate** (address blocks on page 1 of every circular) still ranks in the top 5 for generic questions. Next: strip repeated headers, or detect boilerplate by cross-document frequency.
- **Metadata filtering:** a question naming "payments bank" should filter to that document instead of relying on dedupe.
- **Hybrid search:** combine pgvector with Postgres full-text search for exact terms (circular numbers, "NRD-CSR", "PRAVAAH").
- **Bigger eval set and LLM-as-judge** to catch answers that contain the keyword but are wrong.
- **Model comparison:** re-run the eval with a larger Claude model to measure the accuracy-vs-cost tradeoff.

## Run it

```bash
# 1. Postgres with pgvector
docker run -d --name ragdb -e POSTGRES_PASSWORD=rag -p 5432:5432 pgvector/pgvector:pg16

# 2. Python env
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
echo ANTHROPIC_API_KEY=sk-ant-... > .env

# 3. Put RBI circular PDFs in data/pdfs/ (not committed), then
python ingest.py      # PDFs -> data/chunks.jsonl
python index.py       # embeddings -> Postgres
python ask.py "your question"
python eval.py        # metrics + eval_results.json
```

| File | Role |
|---|---|
| `ingest.py` | Extract, clean and chunk PDFs, with page ranges |
| `embeddings.py` | The only place the embedding model is defined |
| `index.py` | Rebuild the `chunks` table with embeddings |
| `search.py` | Debug tool: raw top-k retrieval with scores |
| `rag.py` | Retrieval, dedupe, two-layer refusal, Claude answer |
| `ask.py` | Demo CLI |
| `eval.py`, `eval_set.json` | Evaluation harness and test questions |

## What I learned

- Retrieval quality is decided before any model runs: extraction, cleaning and chunk boundaries caused more failures than the LLM did.
- Two individually correct components (dedupe and strict grounding) can combine into a wrong answer. Only an end-to-end eval catches that.
- A demo passing is not evidence. v2 fixed the demo and changed nothing on the eval.
- Measure the thing you actually care about: doc-level recall said 100% while answers failed because the right *chunk* wasn't retrieved.
