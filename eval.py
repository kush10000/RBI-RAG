"""M4: measure the system on eval_set.json.

Run:  python eval.py
Writes eval_results.json. Costs ~21 small Claude calls (well under $0.05 on Haiku).

Metrics
  Retrieval (answerable Qs only):
    recall@1 = the #1 retrieved chunk is from the right document
    recall@5 = at least one of the top-5 distinct chunks is from the right document
    Doc-level (not chunk-level) on purpose: it survives re-chunking, so we can compare chunk sizes later.
  Answers:
    answer accuracy  = answerable Q, not refused, and the answer contains an expected keyword
    false refusals   = answerable Q that the system refused
    refusal accuracy = unanswerable Q that the system correctly refused
  Keyword matching is crude but deterministic and transparent. (Alternative: LLM-as-judge.)
"""
import json

from rag import MIN_SCORE, MODEL, TOP_K, answer


def _norm(s: str) -> str:
    return s.lower().replace("’", "'").replace("₹", "rs ")


def _doc_hit(chunk: dict, expected: list[str]) -> bool:
    docs = [chunk["doc"]] + [d for d, _ in chunk["also_in"]]
    return any(e.lower() in d.lower() for d in docs for e in expected)


def main() -> None:
    with open("eval_set.json", encoding="utf-8") as f:
        cases = json.load(f)

    rows = []
    for case in cases:
        r = answer(case["question"])
        row = {"id": case["id"], "answerable": case["answerable"], "top_score": round(r["top_score"], 3),
               "refused": r["refused"], "refused_by": r["refused_by"], "answer": r["answer"]}
        if case["answerable"]:
            row["hit@1"] = bool(r["chunks"]) and _doc_hit(r["chunks"][0], case["expected_docs"])
            row["hit@5"] = any(_doc_hit(c, case["expected_docs"]) for c in r["chunks"][:5])
            row["correct"] = (not r["refused"]) and any(
                k in _norm(r["answer"]) for k in case["expected_keywords"])
            verdict = "PASS" if row["correct"] else ("FALSE REFUSAL" if r["refused"] else "WRONG")
            detail = f"hit@1={int(row['hit@1'])} hit@5={int(row['hit@5'])}"
        else:
            verdict = f"PASS ({r['refused_by']})" if r["refused"] else "ANSWERED (should refuse)"
            detail = ""
        row["verdict"] = verdict
        rows.append(row)
        print(f"{case['id']:<11} score={row['top_score']:.3f}  {detail:<16} {verdict}")

    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} = {100 * n / d:.0f}%" if d else "n/a"

    print(f"\n=== SUMMARY  (model={MODEL}, top_k={TOP_K}, min_score={MIN_SCORE}) ===")
    print(f"Retrieval recall@1   : {pct(sum(r['hit@1'] for r in pos), len(pos))}")
    print(f"Retrieval recall@5   : {pct(sum(r['hit@5'] for r in pos), len(pos))}")
    print(f"Answer accuracy      : {pct(sum(r['correct'] for r in pos), len(pos))}")
    print(f"False refusals       : {pct(sum(r['refused'] for r in pos), len(pos))}")
    print(f"Refusal accuracy     : {pct(sum(r['refused'] for r in neg), len(neg))}")
    print(f"  refused by threshold: {sum(r['refused_by'] == 'threshold' for r in neg)}, "
          f"by LLM: {sum(r['refused_by'] == 'llm' for r in neg)}")
    print(f"Score ranges -> answerable: {min(r['top_score'] for r in pos):.3f}-{max(r['top_score'] for r in pos):.3f} | "
          f"unanswerable: {min(r['top_score'] for r in neg):.3f}-{max(r['top_score'] for r in neg):.3f}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\nWrote eval_results.json (full answers, for reading failures)")


if __name__ == "__main__":
    main()
