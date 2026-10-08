"""M2 part 1: data/chunks.jsonl -> embeddings -> Postgres (pgvector).

Run:  python index.py
Full rebuild every run: drop the table, recreate it, re-insert everything.
"""
import json
import time

import psycopg
from pgvector.psycopg import register_vector

from embeddings import DIM, MODEL_NAME, embed_documents

# Local dev database from the Docker container (user postgres, password rag).
DSN = "postgresql://postgres:rag@localhost:5432/postgres"


def main() -> None:
    with open("data/chunks.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} chunks. Embedding with {MODEL_NAME}...")

    t0 = time.time()
    vectors = embed_documents([r["text"] for r in records])
    print(f"Embedded in {time.time() - t0:.1f}s -> shape {vectors.shape}")

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)  # teaches psycopg to send numpy arrays as pgvector values

        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(f"""
            CREATE TABLE chunks (
                id        TEXT PRIMARY KEY,
                doc       TEXT NOT NULL,
                page      INT  NOT NULL,
                page_end  INT  NOT NULL,
                text      TEXT NOT NULL,
                embedding vector({DIM}) NOT NULL
            )
        """)
        # No vector index (HNSW) yet: with a few hundred rows, an exact full scan takes
        # milliseconds and is 100% accurate. HNSW is approximate; it pays off at ~100k+ rows.

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, doc, page, page_end, text, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                [(r["id"], r["doc"], r["page"], r["page_end"], r["text"], v) for r, v in zip(records, vectors)],
            )

        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"Stored {count} rows in table 'chunks'.")


if __name__ == "__main__":
    main()
