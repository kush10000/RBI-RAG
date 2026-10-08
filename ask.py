"""Demo CLI.

Run:  python ask.py "your question"
"""
import sys

from rag import answer

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who can get one-time approval to buy more shares in a payments bank?"
    result = answer(question)

    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    if result["refused"]:
        print(f"(refused by: {result['refused_by']}, best score {result['top_score']:.3f})\n")

    print("Sources retrieved:")
    for i, c in enumerate(result["chunks"], start=1):
        dupes = "".join(f"\n       same text also in: {d} p.{p}" for d, p in c["also_in"])
        print(f"  [{i}] {c['score']:.3f}  {c['doc']}  p.{c['page']}{dupes}")
    if "tokens" in result:
        print(f"\nTokens: {result['tokens']['in']} in / {result['tokens']['out']} out")
