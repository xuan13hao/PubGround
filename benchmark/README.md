# PubGround benchmark

Every PubGround number here is produced by the PubGround package itself (`LocalCorpus`,
`PubMedSentenceMatcher.rank_abstracts`, `VerdictJudge`); external retrievers, rerankers and verifiers are scored on
the **same candidates and inputs**. Full tables with every metric, 95% bootstrap intervals and paired comparisons:
**[PUBGROUND_RESULTS.md](PUBGROUND_RESULTS.md)** (`results_pubground.json`).

## Run it

```bash
pip install "pubground[bench]"            # or, from a clone: pip install ".[all]"
export PUBGROUND_BENCH_DATA=~/pubground-bench   # where datasets, indexes and predictions go
pubground-bench download                  # SciFact, SciFact-Open, HealthVer + VeriSci / MultiVerS checkpoints
pubground-bench run                       # all systems (GPU; OPENROUTER_API_KEY for the LLM judges)
```

**Compare your own model** on identical inputs (with the released predictions, nothing else needs re-running):

```bash
pubground-bench eval-reranker --model BAAI/bge-reranker-v2-m3 --dataset scifact_open   # any HF cross-encoder
pubground-bench eval-verdict --predictions my_verdicts.jsonl --dataset healthver        # {claim_id, doc_id, label}
```

Both print your score and a paired bootstrap difference against every system below.
`pubground-bench export --out DIR` writes all rankings and verdicts in the release layout (Zenodo / Hugging Face).

## Datasets

| Dataset | Claims | Corpus | Evaluates |
|---|---|---|---|
| SciFact (BEIR test = SciFact dev) | 300 | 5,183 abstracts | retrieval + verdict |
| SciFact-Open | 279 SciFact test claims (206 with evidence) | 500,000 S2ORC abstracts | retrieval + verdict |
| HealthVer (test) | 230 claims, 903 claim–abstract pairs | 322 abstracts | verdict |

## Retrieval (nDCG@10 with 95% CI; Recall@100)

Rerankers score PubGround's hybrid top-100. "PubGround" rows use the package; "single stage" rows are external embedders over the whole corpus.

| System | SciFact nDCG@10 (95% CI) | SciFact R@100 | SciFact-Open nDCG@10 (95% CI) | SciFact-Open R@100 |
|---|---|---|---|---|
| PubGround: BM25 | 0.661 (0.614–0.706) | 0.885 | 0.561 (0.505–0.616) | 0.864 |
| PubGround: MedCPT dense | 0.737 (0.695–0.778) | 0.982 | 0.626 (0.576–0.677) | 0.928 |
| PubGround: hybrid (BM25 + MedCPT dense, RRF) | 0.720 (0.677–0.762) | 0.973 | 0.643 (0.590–0.691) | 0.964 |
| **PubGround: hybrid + MedCPT rerank** | 0.792 (0.753–0.829) | 0.973 | 0.750 (0.707–0.793) | 0.964 |
| PubGround: + LLM query expansion (DeepSeek) | 0.797 (0.760–0.833) | 0.970 | 0.746 (0.702–0.789) | 0.952 |
| hybrid → ms-marco MiniLM-L-12 | 0.690 (0.644–0.734) | 0.973 | 0.639 (0.589–0.690) | 0.964 |
| hybrid → bge-reranker-v2-m3 | 0.734 (0.687–0.776) | 0.973 | 0.737 (0.691–0.779) | 0.964 |
| hybrid → OpenScholar reranker | 0.685 (0.636–0.730) | 0.973 | 0.693 (0.643–0.739) | 0.964 |
| hybrid → monoT5-3B | 0.771 (0.732–0.809) | 0.973 | 0.778 (0.736–0.817) | 0.964 |
| hybrid → Qwen3-Reranker-4B | 0.801 (0.763–0.837) | 0.973 | 0.806 (0.766–0.842) | 0.964 |
| OpenScholar retriever → OpenScholar reranker | 0.696 (0.649–0.739) | 0.963 | 0.651 (0.598–0.703) | 0.877 |
| Qwen3-Embedding-4B (single stage) | 0.784 (0.747–0.818) | 0.977 | 0.688 (0.638–0.740) | 0.944 |
| bge-large-en-v1.5 (single stage) | 0.746 (0.704–0.787) | 0.952 | 0.699 (0.653–0.742) | 0.933 |

**Reranker throughput** (one H100; model loaded and warmed up; median of three runs of 2,000 pairs; PubGround's
MedCPT as the package runs it, fp32 / batch 16):

| Reranker | Pairs/s (H100) |
|---|---|
| MiniLM-L12 | 1621.5 |
| bge-reranker-v2-m3 | 710.5 |
| OpenScholar-Reranker | 498.0 |
| MedCPT (PubGround) | 344.5 |
| monoT5-3B | 116.4 |
| Qwen3-Reranker-4B | 37.9 |

## Verdict (abstract-level label F1 with 95% CI)

Gold = cited abstracts; top-3 = PubGround's three highest-ranked abstracts (hybrid + MedCPT). "—": not comparable
(supervised MultiVerS was trained on the SciFact evaluation claims).

| Verifier | SciFact gold | SciFact top-3 | SciFact-Open top-3 | HealthVer gold |
|---|---|---|---|---|
| PubGround judge, Claude Sonnet 5.5 | 0.873 (0.833–0.909) | 0.664 (0.611–0.714) | 0.466 (0.415–0.524) | 0.514 (0.455–0.573) |
| PubGround judge, DeepSeek V4.1 Flash | 0.833 (0.784–0.877) | 0.646 (0.589–0.699) | 0.470 (0.417–0.530) | 0.481 (0.422–0.539) |
| PubGround judge, Claude Haiku 5.5 | 0.820 (0.771–0.866) | 0.636 (0.581–0.690) | 0.456 (0.405–0.512) | 0.460 (0.401–0.515) |
| PubGround judge, Qwen3-8B (local) | 0.711 (0.654–0.762) | 0.433 (0.388–0.476) | 0.351 (0.315–0.390) | 0.618 (0.568–0.666) |
| VeriSci (supervised, SciFact) | 0.734 (0.670–0.792) | 0.562 (0.507–0.618) | 0.349 (0.293–0.408) | 0.198 (0.148–0.249) |
| MultiVerS (zero-shot) | 0.631 (0.562–0.699) | 0.561 (0.496–0.627) | 0.314 (0.264–0.368) | 0.307 (0.243–0.368) |
| MultiVerS supervised on SciFact (contaminated on SciFact) | — | — | 0.438 (0.374–0.510) | 0.067 (0.040–0.095) |
| MultiVerS supervised on HealthVer | — | — | — | 0.777 (0.729–0.818) |

## Findings

1. **Hybrid retrieval** raises Recall@100 over BM25 on both corpora (0.885 → 0.973; 0.864 → 0.964) and is the best
   first stage at 500,000 abstracts; on SciFact, MedCPT dense alone is marginally higher (0.982).
2. **PubGround's MedCPT reranking** matches the best reranker on SciFact (0.792 vs 0.801 for Qwen3-Reranker-4B;
   paired difference −0.009, CI −0.032 to 0.014) at 9× its throughput, but Qwen3-Reranker-4B is better at
   500,000 abstracts (0.806 vs 0.750; −0.056, −0.086 to −0.025).
3. **OpenScholar.** PubGround exceeds OpenScholar's retriever → reranker pipeline by 0.096 (0.064–0.130) on SciFact and
   0.099 (0.062–0.137) on SciFact-Open. OpenScholar's reranker scores below its own base model, bge-reranker-large, on
   the same candidates (0.685 vs 0.742; 0.693 vs 0.720), consistent with its tuning on question-style queries.
4. **LLM query expansion** does not change nDCG@10 once MedCPT reranking is applied (all differences within ±0.005,
   every interval includes 0).
5. **PubGround's judge** (Claude / DeepSeek) beats every verifier not trained on the target dataset, including VeriSci on
   its own SciFact data (+0.086 to +0.139), and is comparable to supervised MultiVerS on SciFact-Open (+0.018 to +0.032,
   every interval includes 0). On HealthVer the in-domain supervised MultiVerS wins (+0.262 to +0.317 over the API
   judges): the API judges answer NEI on 52–57% of HealthVer evidence pairs vs 7–12% on SciFact, consistent with
   HealthVer counting support for one part of a multi-part claim as SUPPORT.

## Caveats

- **Contamination.** The released MultiVerS `scifact` checkpoint was trained on all 300 SciFact dev claims (the BEIR
  SciFact test claims) and scores 100% on them; it is compared only on SciFact-Open.
- **Reproduction gaps.** VeriSci reproduces at 0.734 oracle label F1 vs 0.774 reported (precision gap unresolved);
  the MultiVerS port reproduces its HealthVer result (0.777).
- **SciFact-Open labels come from pooling** (4 older systems), which lowers every system's precision.
- **Scope.** The benchmark scores abstracts on fixed corpora with local MedCPT dense retrieval; live PubGround
  searches PubMed through E-utilities and LitSense 2.0 and returns sentences.
- **Ties.** Qwen3-Reranker-4B scores (bf16 log-probabilities) saturate; tie-breaking changes its nDCG@10 by ≤0.006.
- **Cost.** All LLM calls for these experiments cost US$8.56 (Claude Sonnet 5.5: US$7.91); responses are cached in
  `data/llm_cache/`, identical prompts are sent once, and every table is regenerated from the cache.

## Files

| File | Role |
|---|---|
| `run_pubground.py` | PubGround systems, external rerankers, OpenScholar, throughput, verdicts (via the package API) |
| `pubground_report.py` | metrics, bootstrap intervals, paired tests → `results_pubground.json`, `PUBGROUND_RESULTS.md` |
| `cli.py` | `pubground-bench` command line |
| `export_release.py` | release layout of all predictions (Zenodo / Hugging Face) |
| `RESULTS.md`, `results_*.json`, `run_llm_benchmark.py`, `run_baselines_*.py`, `run_extra_datasets.py` | earlier harness (before PubGround's retrieval was refactored into the package); kept for reference |
