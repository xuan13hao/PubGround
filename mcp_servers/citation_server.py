"""
Citation Matcher MCP Server (official MCP SDK / FastMCP).

Exposes the PubMed sentence-matching / citation-grounding system as MCP tools so
that any MCP client (Claude Code, Claude Desktop, other agents) can call it to
ground a sentence in the biomedical literature.

Replaces the earlier hand-rolled JSON-RPC server. Built on the official `mcp`
SDK, which handles protocol negotiation, the initialize handshake, notifications,
and transport.

stdio hygiene: under stdio transport, anything written to stdout corrupts the
JSON-RPC stream. The matcher pipeline uses print() for progress, so every tool
call redirects stdout to stderr for its duration.

Run (stdio):
    python -m mcp_servers.citation_server
    # or
    python mcp_servers/citation_server.py

Register with Claude Code:
    claude mcp add citation-matcher -- python /abs/path/mcp_servers/citation_server.py
"""

import contextlib
import os
import sys

# Ensure the package is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from pubmed_sentence_matcher import PubMedSentenceMatcher
from pubmed_sentence_matcher.config import Config

mcp = FastMCP("citation-matcher")

# Single lazily-created matcher so the heavy reranker model loads at most once.
_matcher: PubMedSentenceMatcher | None = None


def _get_matcher() -> PubMedSentenceMatcher:
    global _matcher
    if _matcher is None:
        use_reranker = os.getenv("CITATION_USE_RERANKER", "1") not in ("0", "false", "False")
        _matcher = PubMedSentenceMatcher(
            email=os.getenv("PUBMED_EMAIL", "citation-matcher@example.com"),
            config=Config(),
            similarity_threshold=float(os.getenv("CITATION_SIM_THRESHOLD", "0.5")),
            use_reranker=use_reranker,
            # second first-stage retriever fused by RRF: LitSense (MedCPT dense over PubMed)
            dense_recall=None if os.getenv("CITATION_DENSE_RECALL", "litsense") in ("0", "none", "None") else "litsense",
        )
    return _matcher


def _trim_result(r: dict) -> dict:
    """Shrink a pipeline result into a compact, JSON-friendly citation record."""
    article = r.get("article", {}) or {}
    scores = r.get("scores", {}) or {}
    return {
        "sentence": r.get("sentence", ""),
        "citation": r.get("citation", ""),
        "citation_format": r.get("citation_format", ""),
        "article": {
            "pmid": article.get("pmid", ""),
            "title": article.get("title", ""),
            "journal": article.get("journal", ""),
            "year": article.get("pub_year", article.get("year", "")),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid', '')}/"
            if article.get("pmid")
            else "",
        },
        "scores": {
            "final_rank": round(r.get("final_rank_score", 0.0), 4),
            "rerank": round(scores.get("rerank", 0.0), 4),
            "semantic": round(scores.get("semantic", 0.0), 4),
            "multi_signal": round(scores.get("final", 0.0), 4),
        },
        **({"verdict": {k: r["verdict"][k] for k in ("label", "rationale_text")}} if r.get("verdict") else {}),
    }


@mcp.tool()
async def find_citations(
    sentence: str,
    top_k: int = 10,
    citation_format: str = "mla",
    max_articles: int = 30,
    verify: bool = False,
) -> dict:
    """
    Find the best PubMed sentences to cite for a given claim/sentence.

    Runs the full pipeline: LLM query expansion -> PubMed E-utilities search plus
    LitSense dense recall, fused by reciprocal rank fusion -> MedCPT cross-encoder
    reranking of abstracts and sentences -> ranking, then formats a citation for each
    match. With verify=True each cited abstract is also labelled SUPPORT /
    CONTRADICT / NEI by an LLM judge, with its rationale sentences.

    Args:
        sentence: The input claim/sentence to ground in the literature.
        top_k: Number of citations to return (default 10).
        citation_format: One of "mla", "apa", "nature" (default "mla").
        max_articles: Max PubMed articles to process (default 30; higher = slower
            but better recall).
        verify: Label each result's abstract SUPPORT / CONTRADICT / NEI (one LLM
            call per distinct abstract; default False).

    Returns:
        {"query": str, "count": int, "results": [ {sentence, citation, article,
        scores, verdict?}, ... ]}
    """
    matcher = _get_matcher()
    with contextlib.redirect_stdout(sys.stderr):
        results = await matcher.find_matching_sentences(
            input_sentence=sentence,
            top_k=top_k,
            max_articles=max_articles,
            citation_format=citation_format,
            verify=verify,
        )
    return {
        "query": sentence,
        "count": len(results),
        "results": [_trim_result(r) for r in results],
    }


@mcp.tool()
async def pubmed_search(query: str, max_results: int = 20) -> dict:
    """
    Search PubMed and return matching article PMIDs for a raw query string.

    A low-level building block (no reranking / grounding). Use `find_citations`
    for end-to-end citation grounding.

    Args:
        query: PubMed search query string.
        max_results: Max PMIDs to return (default 20).

    Returns:
        {"query": str, "pmids": [str, ...]}
    """
    matcher = _get_matcher()
    with contextlib.redirect_stdout(sys.stderr):
        pmids = await matcher._search_pubmed_with_query(query, max_results=max_results)
    return {"query": query, "pmids": pmids}


def main() -> None:
    """Console entry point (``pubground-mcp``); stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
