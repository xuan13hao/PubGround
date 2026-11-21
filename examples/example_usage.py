"""
Example: ground a sentence in the PubMed literature with reranking.

Run:
    cd examples
    python example_usage.py

Requires OPENAI_API_KEY (query optimization + LLM verification + embeddings).
The first run downloads the MedCPT cross-encoder (~400MB) for reranking; set
use_reranker=False to skip it.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pubmed_sentence_matcher import PubMedSentenceMatcher
from pubmed_sentence_matcher.config import Config


async def main():
    matcher = PubMedSentenceMatcher(
        email=os.getenv("PUBMED_EMAIL", "you@example.com"),
        config=Config(),
        similarity_threshold=0.5,
        use_reranker=True,  # MedCPT cross-encoder (Stage 4.5)
    )

    sentence = "Machine learning models can accurately predict patient outcomes."
    results = await matcher.find_matching_sentences(
        input_sentence=sentence,
        top_k=5,
        max_articles=30,
        citation_format="mla",
    )

    print(f"\n=== {len(results)} citations for: {sentence} ===\n")
    for i, r in enumerate(results, 1):
        s = r["scores"]
        print(f"[{i}] rerank={s.get('rerank', 0):.3f}  "
              f"multi_signal={s['final']:.3f}  "
              f"final_rank={r.get('final_rank_score', 0):.3f}")
        print(f"    match:    {r['sentence'][:100]}")
        print(f"    citation: {r['citation']}\n")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved full results to results.json")


if __name__ == "__main__":
    asyncio.run(main())
