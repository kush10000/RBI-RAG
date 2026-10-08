"""M1 (v3): PDF -> clean text -> chunks with metadata -> data/chunks.jsonl

Run:  python ingest.py

v3 change: chunk the WHOLE DOCUMENT instead of page by page.
Eval showed answers that straddle a page break (e.g. "...report ... five per cent of the |page 5| banking
company ... within three working days") were split into two chunks, and neither chunk alone answered
the question. Now chunks flow across page breaks and each chunk records the page range it covers.
"""
import json
import re
from bisect import bisect_right
from pathlib import Path

from pypdf import PdfReader

PDF_DIR = Path("data/pdfs")
OUT = Path("data/chunks.jsonl")

# bge-small reads at most 512 tokens; anything longer is SILENTLY truncated.
# ~1200 chars is roughly 250-300 tokens, so every chunk fits with room to spare.
CHUNK_CHARS = 1200
# Overlap so a sentence cut at a chunk boundary still appears whole in one chunk.
OVERLAP = 200

# RBI PDFs are bilingual. bge-small-en is English-only: Hindi becomes noise in the vector.
# � is the "replacement character" pypdf emits for Hindi glyphs it cannot decode.
DEVANAGARI = re.compile(r"[ऀ-ॿ�]+")
# Form-fill blanks (_____) and table-of-contents dot leaders (.......) carry no meaning.
LEADERS = re.compile(r"[_.\-–—]{4,}")
# Chunks shorter than this are stray headers, not content.
MIN_CHUNK_CHARS = 50


def clean(text: str) -> str:
    text = text.replace(" ", " ")          # non-breaking spaces -> normal spaces
    text = DEVANAGARI.sub("", text)              # drop Hindi + undecodable glyphs
    text = LEADERS.sub(" ", text)                # drop ____ and ..... runs
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    lines = [ln.strip() for ln in text.splitlines()]
    # Drop lines with no letters/digits (debris after Hindi removal) and bare page numbers ("5").
    lines = [ln for ln in lines if not ln or (any(c.isalnum() for c in ln) and not ln.isdigit())]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse big vertical gaps
    return text.strip()


def chunk_spans(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """(start, end) character offsets of fixed-size, overlapping chunks, nudged to end at a
    line or sentence boundary. Offsets (not strings) so we can map each chunk back to its pages."""
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer to cut at the last newline or ". " inside the window,
            # but only if that keeps the chunk at least half full.
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        spans.append((start, end))
        if end >= len(text):
            break
        start = end - overlap  # step back so consecutive chunks share `overlap` chars
    return spans


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR.resolve()}")
        return

    records = []
    for pdf in pdfs:
        reader = PdfReader(pdf)

        # Join all pages into one string, remembering where each page starts.
        parts, page_starts, page_numbers, offset = [], [], [], 0
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean(page.extract_text() or "")
            if not text:
                continue  # scanned/image-only page: no text layer
            page_starts.append(offset)
            page_numbers.append(page_no)
            parts.append(text)
            offset += len(text) + 2  # +2 for the "\n\n" separator below
        full = "\n\n".join(parts)

        def page_at(pos: int) -> int:
            return page_numbers[bisect_right(page_starts, pos) - 1]

        n_doc = 0
        for start, end in chunk_spans(full):
            chunk = full[start:end].strip()
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            records.append({
                "id": f"{pdf.stem}-c{n_doc}",
                "doc": pdf.name,
                "page": page_at(start),             # page where the chunk starts
                "page_end": page_at(max(start, end - 1)),  # page where it ends
                "text": chunk,
            })
            n_doc += 1
        print(f"{pdf.name}: {len(reader.pages)} pages -> {n_doc} chunks")

    # encoding="utf-8" matters on Windows: the default (cp1252) crashes on characters like the rupee sign.
    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if records:
        lengths = [len(r["text"]) for r in records]
        spanning = sum(r["page"] != r["page_end"] for r in records)
        print(f"\nTOTAL: {len(records)} chunks | avg {sum(lengths) // len(lengths)} chars | "
              f"min {min(lengths)} | max {max(lengths)} | {spanning} span a page break")
        print(f"Wrote {OUT}")
    else:
        print("No text extracted. Are these scanned PDFs?")


if __name__ == "__main__":
    main()
