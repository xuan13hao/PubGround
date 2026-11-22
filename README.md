# PubGround

*Formerly PubMed Sentence Matcher. The Python package keeps its import name, `pubmed_sentence_matcher`.*

**Ground any claim in the biomedical literature.** Give the system one sentence; it searches PubMed, retrieves candidate evidence, reranks it with a biomedical cross-encoder, and returns the exact published sentences you can cite — each with a formatted citation (MLA / APA / Nature).

🔬 **Live demo & overview:** [xuan13hao.github.io/citation_agents_system](https://xuan13hao.github.io/citation_agents_system/)

> The task behind citation recommendation, claim attribution, and RAG faithfulness: a sentence can be *similar* to your claim without actually *supporting* it. A cheap first-pass similarity lets those false positives through — so a sharper second pass (a cross-encoder reranker) does the deciding.

---

## Pipeline

The engine runs a multi-stage pipeline. The multi-signal similarity is a cheap first pass that shortlists candidates; the **MedCPT cross-encoder (Stage 4.5)** is the primary relevance signal.

| # | Stage | Role |
|---|-------|------|
| 1 | **LLM query optimization** | Expand the claim into 3–5 PubMed queries (synonyms, MeSH-aware phrasing) |
| 2 | **Multi-strategy PubMed search** | Run every query via NCBI E-utilities, merge PMIDs |
| 3 | **Abstract retrieval + sentence split** | Fetch abstracts, segment into candidate sentences with context |
| 4 | **Multi-signal similarity** | Semantic + keyword shortlist — cheap first pass, *not* the verdict |
| 4.5 | **MedCPT cross-encoder rerank** ⭐ | Jointly score `(claim, passage)` pairs — the primary relevance signal |
| 5 | **Final ranking + dedup** | Blend rerank × article quality; one best hit per article |

### Why the reranker matters

A controlled probe against *"Machine learning models can accurately predict patient outcomes."*:

| Candidate sentence | First-pass | After MedCPT |
|---|---|---|
| The restaurant served excellent pasta on Tuesday evenings. | **0.80** (ranks #1 ✗) | **0.00** (demoted ✓) |
| Random forest classifiers improved prognosis prediction in oncology cohorts. | 0.55 | **1.00** |
| Deep learning achieved high accuracy predicting ICU patient outcomes. | 0.52 | **1.00** |

The first-pass score ranks a decoy top by surface overlap; the cross-encoder drives it to zero and promotes the real evidence.

---

## Features

- **MedCPT cross-encoder reranking** — biomedical-domain SOTA reranker ([NCBI MedCPT](https://github.com/ncbi/MedCPT), a PubMedBERT cross-encoder trained on 255M PubMed click logs). Runs on GPU if available, and degrades gracefully to the multi-signal score if `torch`/`transformers` or the model are unavailable.
- **MCP server** — exposes the whole system as [Model Context Protocol](https://modelcontextprotocol.io) tools (official SDK / FastMCP) so Claude Code, Claude Desktop, and other agents can call it directly.
- **Multiple citation formats** — MLA, APA, and Nature.
- **Multi-signal first pass** — semantic embeddings + keyword/structure/context, used to shortlist candidates for the reranker.
- **LLM-enhanced** — LLM for query optimization (one call per query).
- **Standalone** — no dependency on any external research framework.

---

## Installation

Requires **Python 3.10+**.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download NLTK data for better sentence tokenization
python -c "import nltk; nltk.download('punkt')"

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

> **Reranker note:** `transformers` must match your installed `torch`. For `torch` 2.1–2.5 use `transformers>=4.44,<5` (as tested); for `torch>=2.6` use `transformers>=5`. The MedCPT model (~400 MB) downloads automatically on first use. GPU is optional but strongly recommended.

---

## Quick Start

```python
import asyncio
from pubmed_sentence_matcher import PubMedSentenceMatcher
from pubmed_sentence_matcher.config import Config

async def main():
    matcher = PubMedSentenceMatcher(
        email="your-email@example.com",
        config=Config(),
        similarity_threshold=0.5,
        use_reranker=True,   # MedCPT cross-encoder (Stage 4.5)
    )

    results = await matcher.find_matching_sentences(
        input_sentence="Machine learning models can accurately predict patient outcomes.",
        top_k=10,
        max_articles=50,
        citation_format="mla",
    )

    for r in results:
        print(f"[{r['scores'].get('rerank', 0):.3f}] {r['sentence']}")
        print(f"         {r['citation']}\n")

asyncio.run(main())
```

### Runnable examples

```bash
cd examples
python example_usage.py            # full pipeline (needs OPENAI_API_KEY)
python retrieval_rerank_demo.py    # retrieval + MedCPT rerank only — no API key needed
```

---

## MCP Server

Rebuilt on the official MCP SDK (FastMCP), [`mcp_servers/citation_server.py`](mcp_servers/citation_server.py) exposes two tools:

- **`find_citations(sentence, top_k, citation_format, max_articles)`** — end-to-end citation grounding
- **`pubmed_search(query, max_results)`** — low-level PMID search

Run it (stdio transport):

```bash
python -m mcp_servers.citation_server
```

Register with Claude Code:

```bash
claude mcp add citation-matcher -- python /abs/path/to/mcp_servers/citation_server.py
```

Environment toggles: `CITATION_USE_RERANKER=0` disables reranking · `CITATION_SIM_THRESHOLD` sets the first-pass threshold · `PUBMED_EMAIL` sets the NCBI identification email.

> The server redirects pipeline stdout to stderr so the stdio JSON-RPC stream stays clean. The earlier hand-rolled JSON-RPC files (`mcp_servers/pubmed_server.py`, `pubmed_sentence_matcher/mcp_client.py`) are superseded by this.

---

## Configuration

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Used for query optimization and embeddings |
| `PUBMED_EMAIL` | — | Email for NCBI API identification (recommended) |
| `NCBI_API_KEY` | — | Raises E-utilities rate limit from 3 → 10 req/s |
| `OPENAI_BASE_URL` | — | Custom endpoint for OpenAI-compatible APIs |
| `EMBEDDING` | — | Embedding model (default `openai:text-embedding-3-small`) |
| `SMART_LLM` | — | LLM model (default `openai:gpt-4-turbo`) |

### Reranker parameters

```python
matcher = PubMedSentenceMatcher(
    email="you@example.com",
    config=Config(),
    use_reranker=True,                        # Stage 4.5 on/off
    reranker_model="ncbi/MedCPT-Cross-Encoder",
    rerank_candidates=50,                     # top-N shortlisted for the cross-encoder
    reranker_device=None,                     # None = auto (cuda if available)
    # first-pass multi-signal weights:
    similarity_threshold=0.5,
    semantic_weight=0.5, keyword_weight=0.25,
    structure_weight=0.15, context_weight=0.1,
)
```

---

## Output Format

Each result is a dict containing:

- `sentence` — the matched sentence text
- `article` — metadata (title, authors, journal, year, PMID)
- `scores` — `rerank` (normalized 0–1), `rerank_raw` (logit), `primary` (score the final ranking is built on), plus the first-pass `semantic` / `keyword` / `structure` / `context` / `final`
- `citation` — formatted citation (MLA / APA / Nature)
- `context` — surrounding sentences from the article

---

## Project Structure

```
citation_agents_system/
├── pubmed_sentence_matcher/
│   ├── pubmed_sentence_matcher.py  # Main matcher (the multi-stage pipeline)
│   ├── reranker.py                 # MedCPT cross-encoder reranker (Stage 4.5)
│   ├── pubmed_citation.py          # Citation formatting (MLA / APA / Nature)
│   ├── config.py                   # Configuration
│   ├── llm_utils.py                # LLM utilities
│   └── embeddings.py               # Embedding utilities
├── mcp_servers/
│   └── citation_server.py          # Official MCP (FastMCP) server
├── examples/
│   ├── example_usage.py            # Full pipeline example
│   └── retrieval_rerank_demo.py    # Key-free retrieval + rerank demo
├── docs/                           # GitHub Pages promo site
├── requirements.txt
├── .env.example
└── README.md
```

---

## Background

The design follows the current state of the art for scientific citation grounding, which converges on a four-stage shape — **retrieve → fuse → cross-encoder rerank → entailment-check**:

- **Cross-encoder reranking** is the single largest quality gain reported (+17.2pp MRR@3, +12.1pp Recall@5 over unreranked retrieval) — see [*How Good are LLM-based Rerankers?*](https://aclanthology.org/2025.findings-emnlp.305.pdf) (EMNLP Findings 2025).
- **MedCPT** sets SOTA on biomedical IR zero-shot — [Jin et al., *Bioinformatics* 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10627406/).
- Full grounding verdicts (SUPPORTS / REFUTES / NEI) build on [SciFact](https://www.alphaxiv.org/abs/2004.14974) and the attribution literature.

### Roadmap

- **Dense retrieval + RRF** — add MedCPT's query encoder alongside keyword search, fused with Reciprocal Rank Fusion, to raise recall at Stage 2.
- **NLI entailment verdict** — upgrade "similar" to SUPPORTS / REFUTES / NEI with rationale spans (NLI → LLM-judge cascade).
- **PMC full text** — wire the BioC API for open-access full text, escalating from abstract to passage-level evidence.

---

## Requirements

- Python 3.10+
- OpenAI API key (for query optimization and embeddings)
- Internet connection (PubMed / NCBI E-utilities)
- `torch` + `transformers` for reranking (GPU recommended; optional — falls back gracefully)

---

## ✍️ Citation
If you have obtained permission and are using this software in your project, please cite the software repository as follows:
**Software Citation (BibTeX):**
```bibtex
@software{pubground_2026,
  author = {Xuan, Hao},
  title = {PubGround: training-free grounding of biomedical claims in PubMed},
  year = {2026},
  url = {https://github.com/xuan13hao/citation_agents_system}
}
```
## Copyright (c) 2026 Hao Xuan. All Rights Reserved.
This software and its associated documentation files are proprietary. Explicit written permission from the author is required for any use, reproduction, modification, or distribution of this code, whether for commercial or non-commercial purposes.
To request permission, please contact:xuan13hao@gmail.com
