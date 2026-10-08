"""M2 part 2: question -> top-k most similar chunks.

Run:  python search.py "your question here"
"""
import sys

import psycopg
from pgvector.psycopg import register_vector

from embeddings import embed_query
from index import DSN


def search(question: str, k: int = 5) -> list[tuple]:
    qvec = embed_query(question)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        # <=> is pgvector's COSINE DISTANCE operator (0 = same direction).
        # ORDER BY distance ascending = most similar first.
        # 1 - distance = cosine similarity, the same number your test_embed.py matrix showed.
        return conn.execute(
            """
            SELECT doc, page, text, 1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, qvec, k),
        ).fetchall()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who can hold shares in a payments bank?"
    print(f"Q: {question}\n")
    for rank, (doc, page, text, score) in enumerate(search(question), start=1):
        preview = text[:250].replace("\n", " ")
        print(f"#{rank}  score={score:.3f}  {doc}  p.{page}\n    {preview}...\n")
