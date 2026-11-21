# PubMed Sentence Matcher

A standalone, high-accuracy sentence matching system that finds the most relevant sentences in PubMed articles for any given input sentences. 
## Features

- **7-Stage Processing Pipeline**: LLM query optimization → Multi-strategy search → Abstract retrieval → Multi-signal similarity (first pass) → **Cross-encoder reranking (MedCPT)** → LLM verification → Final ranking
- **MedCPT Cross-Encoder Reranking**: Biomedical-domain SOTA reranker (NCBI, PubMedBERT cross-encoder trained on 255M PubMed click logs). Jointly scores (query, passage) pairs — the single largest precision gain in modern retrieval pipelines. Runs on GPU if available, degrades gracefully if `torch`/`transformers` are missing.
- **Multi-Signal Similarity**: Cheap first-pass shortlist (semantic embeddings + keyword/structure/context) feeding the reranker
- **LLM-Enhanced**: Uses LLM for query optimization and result verification
- **Multiple Citation Formats**: Supports MLA, APA, and Nature citation styles
- **MCP Server**: Exposes the whole system as [Model Context Protocol](https://modelcontextprotocol.io) tools (official SDK / FastMCP) so Claude Code, Claude Desktop, and other agents can call it directly
- **Standalone**: No dependencies on GPT-Researcher framework

## Installation

1. Clone or copy this project folder

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (optional, for better sentence tokenization):
```bash
python -c "import nltk; nltk.download('punkt')"
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Or set environment variables directly:
```bash
export OPENAI_API_KEY='your-api-key-here'
export PUBMED_EMAIL='your-email@example.com'
```

## Quick Start

```python
import asyncio
from pubmed_sentence_matcher import PubMedSentenceMatcher
from pubmed_sentence_matcher.config import Config

async def main():
    config = Config()
    matcher = PubMedSentenceMatcher(
        email="your-email@example.com",
        config=config,
        similarity_threshold=0.5,
    )
    
    results = await matcher.find_matching_sentences(
        input_sentence="Machine learning models can accurately predict patient outcomes.",
        top_k=10,
        max_articles=50,
        citation_format="mla"
    )
    
    for result in results:
        print(f"Match: {result['sentence']}")
        print(f"Score: {result['scores']['final']:.3f}")
        print(f"Citation: {result['citation']}\n")

asyncio.run(main())
```

## Run Example

```bash
cd examples
python example_usage.py
```

The results will be saved to a JSON file automatically.

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `PUBMED_EMAIL` (optional): Email for PubMed API identification
- `OPENAI_BASE_URL` (optional): Custom base URL for OpenAI-compatible APIs
- `EMBEDDING` (optional): Embedding model, default: `openai:text-embedding-3-small`
- `SMART_LLM` (optional): LLM model, default: `openai:gpt-4-turbo`

### Matcher Parameters

```python
matcher = PubMedSentenceMatcher(
    email="your-email@example.com",
    config=config,
    similarity_threshold=0.5,      # Minimum similarity (0-1)
    semantic_weight=0.5,           # Semantic similarity weight
    keyword_weight=0.25,           # Keyword overlap weight
    structure_weight=0.15,         # Structure similarity weight
    context_weight=0.1,            # Context relevance weight
)
```

## Output Format

Each result contains:
- `sentence`: The matched sentence text
- `article`: Article metadata (title, authors, journal, year, PMID)
- `scores`: Similarity scores (semantic, keyword, structure, context, final)
- `llm_verification`: LLM verification with relevance score and reasoning
- `citation`: Formatted citation (MLA/APA/Nature)
- `context`: Sentence context from the article

## Project Structure

```
pubmed_sentence_matcher_project/
├── pubmed_sentence_matcher/
│   ├── __init__.py
│   ├── pubmed_sentence_matcher.py  # Main matcher class
│   ├── pubmed_citation.py          # Citation formatting
│   ├── config.py                   # Configuration
│   ├── llm_utils.py                # LLM utilities
│   └── embeddings.py               # Embedding utilities
├── examples/
│   └── example_usage.py            # Example script
├── requirements.txt
├── .env.example
└── README.md
```

## MCP Server

The system is exposed as an MCP server (official SDK / FastMCP) in
[`mcp_servers/citation_server.py`](mcp_servers/citation_server.py), with two tools:

- `find_citations(sentence, top_k, citation_format, max_articles)` — end-to-end citation grounding
- `pubmed_search(query, max_results)` — low-level PMID search

Run it (stdio transport):

```bash
python -m mcp_servers.citation_server
```

Register with Claude Code:

```bash
claude mcp add citation-matcher -- python /abs/path/to/mcp_servers/citation_server.py
```

Environment toggles: `CITATION_USE_RERANKER=0` disables reranking,
`CITATION_SIM_THRESHOLD` sets the first-pass threshold, `PUBMED_EMAIL` sets the
NCBI identification email.

> Note: the server redirects pipeline stdout to stderr so the stdio JSON-RPC
> stream stays clean. The earlier hand-rolled JSON-RPC implementation
> (`mcp_servers/pubmed_server.py`, `mcp_client.py`) is superseded by this.

## Reranking (MedCPT)

Reranking is on by default. Control it via the matcher constructor:

```python
matcher = PubMedSentenceMatcher(
    email="you@example.com",
    config=Config(),
    use_reranker=True,                       # Stage 4.5
    reranker_model="ncbi/MedCPT-Cross-Encoder",
    rerank_candidates=50,                    # top-N shortlisted for the cross-encoder
    reranker_device=None,                    # None = auto (cuda if available)
)
```

Each result's `scores` dict gains `rerank` (normalized 0-1), `rerank_raw` (logit),
and `primary` (the score the final ranking is built on). If `torch`/`transformers`
or the model are unavailable, the pipeline silently falls back to the multi-signal
score.

## How It Works

1. **Stage 1**: LLM generates optimized PubMed search queries from input sentence
2. **Stage 2**: Multi-strategy PubMed search using generated queries
3. **Stage 3**: Fetch article abstracts, split into sentences
4. **Stage 4**: Multi-signal similarity (semantic + keyword + structure + context) — cheap first-pass shortlist
5. **Stage 4.5**: **MedCPT cross-encoder reranking** of the shortlist — primary relevance signal
6. **Stage 5**: LLM verifies matches for accuracy and relevance
7. **Stage 6**: Final ranking with deduplication and quality scoring

## ✍️ Citation
If you have obtained permission and are using this software in your project, please cite the software repository as follows:
**Software Citation (BibTeX):**
```bibtex
@software{pubmed_sentence_matcher_2026,
  author = {Xuan, Hao},
  title = {PubMed Sentence Matcher: A multi-stage high-accuracy sentence matching system for biomedical literature},
  year = {2026},
  url = {https://github.com/xuan13hao/citation_agents_system}
}
```
## Copyright (c) 2026 Hao Xuan. All Rights Reserved.
This software and its associated documentation files are proprietary. Explicit written permission from the author is required for any use, reproduction, modification, or distribution of this code, whether for commercial or non-commercial purposes.
To request permission, please contact:xuan13hao@gmail.com

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection (for PubMed API)


