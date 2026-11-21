# Benchmark — SciFact (BEIR)

Claim → evidence retrieval on the standard biomedical claim-grounding benchmark:
**300 test claims** over a **5,183-abstract** corpus. No OpenAI key required — this
isolates retrieval + reranking quality.

## Systems compared

| System | What it is |
|---|---|
| BM25 | Lexical baseline |
| MedCPT dense | MedCPT bi-encoder (query + article encoders) over the full corpus |
| BM25 + MedCPT rerank | BM25 top-100 reranked by the MedCPT cross-encoder (this project's Stage 4.5) |

## Metrics (BEIR-standard + extras)

`nDCG@10`, `Recall@10`, `Recall@100`, `MRR@10`, `P@1`, macro-averaged over queries.

## Results (300 queries, topk=100)

| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | P@1 |
|---|---|---|---|---|---|
| BM25 | 0.652 | 0.776 | 0.873 | 0.618 | 0.523 |
| **BM25 + MedCPT rerank** | **0.759** | 0.855 | 0.873 | **0.734** | **0.660** |
| MedCPT dense | 0.737 | 0.876 | **0.982** | 0.701 | 0.607 |

**Reranking adds +0.106 nDCG@10 (+16.3%) over BM25** — matching the field's reported
~+17pp cross-encoder gain. The BM25 baseline (0.652) matches the published 0.665,
confirming the harness. Note reranking's `Recall@100` equals BM25's by construction
(it only reorders BM25's top-100); dense retrieval wins recall because it searches
the whole corpus. See `results.json` for the exact run.

## Reproduce

```bash
# 1. Install deps
pip install rank_bm25 pandas pyarrow huggingface_hub transformers torch

# 2. Download the dataset (BEIR SciFact via HuggingFace)
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
for repo, fn, local in [
    ("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", "data/corpus.parquet"),
    ("BeIR/scifact", "queries/queries-00000-of-00001.parquet", "data/queries.parquet"),
    ("BeIR/scifact-qrels", "test.tsv", "data/qrels_test.tsv"),
]:
    shutil.copy(hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset"), local)
PY

# 3. Run
python run_benchmark.py --topk 100            # full 300-query run
python run_benchmark.py --max-queries 20 --skip-dense   # quick smoke test
```

GPU strongly recommended (the dense and rerank stages run transformer encoders).
Full run ≈ 11 min on an RTX 3060.
