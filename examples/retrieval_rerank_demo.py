"""Real end-to-end test: PubMed retrieval + MedCPT rerank (no OpenAI key needed).

Uses a single batched efetch to avoid NCBI rate limits (429) when no API key is set.
"""
import asyncio, os, json, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-construct-only")

from pubmed_sentence_matcher.pubmed_sentence_matcher import PubMedSentenceMatcher
from pubmed_sentence_matcher.config import Config

CLAIM = "Statins reduce the risk of cardiovascular events in patients with high cholesterol."
EMAIL = "test@example.com"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def esearch(term, n):
    url = f"{BASE}/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": term, "retmax": n, "retmode": "json", "email": EMAIL})
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read()).get("esearchresult", {}).get("idlist", [])


def efetch_batch(pmids):
    """One request for all PMIDs -> list of (pmid, title, journal, year, [sentences])."""
    url = f"{BASE}/efetch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "email": EMAIL})
    with urllib.request.urlopen(url) as r:
        root = ET.fromstring(r.read())
    arts = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        title = (art.findtext(".//ArticleTitle") or "").strip()
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = (art.findtext(".//PubDate/Year") or "").strip()
        abstract = " ".join(t.text or "" for t in art.findall(".//AbstractText")).strip()
        arts.append({"pmid": pmid, "title": title, "journal": journal, "year": year,
                     "abstract": abstract})
    return arts


async def main():
    m = PubMedSentenceMatcher(email=EMAIL, config=Config(), use_reranker=True)

    pmids = esearch("statins cardiovascular events cholesterol", 15)
    print(f"PubMed returned {len(pmids)} PMIDs")
    time.sleep(0.5)
    arts = efetch_batch(pmids)
    print(f"Fetched {len(arts)} articles in one batch efetch")

    # Split abstracts into candidate sentences
    cands = []
    for a in arts:
        for s in m.sent_tokenize(a["abstract"]):
            s = s.strip()
            if len(s) >= 25:
                cands.append({"text": s, "article": a})
    print(f"{len(cands)} candidate sentences")

    # MedCPT cross-encoder rerank over all real candidates
    scores = await m.reranker.score_normalized(CLAIM, [c["text"][:1000] for c in cands])
    ranked = sorted(zip(cands, scores), key=lambda x: -x[1])

    out = []
    print(f"\n=== Top citations for claim: {CLAIM} ===\n")
    for c, s in ranked[:6]:
        a = c["article"]
        rec = {"score": round(float(s), 4), "sentence": c["text"],
               "title": a["title"], "journal": a["journal"], "year": a["year"], "pmid": a["pmid"]}
        out.append(rec)
        print(f"[{rec['score']:.4f}] {rec['sentence'][:115]}")
        print(f"         {rec['journal']} ({rec['year']}) PMID:{rec['pmid']}\n")

    with open("scratch_e2e_results.json", "w") as f:
        json.dump({"claim": CLAIM, "n_pmids": len(pmids), "n_articles": len(arts),
                   "n_candidates": len(cands), "top": out}, f, indent=2)
    print("Saved -> scratch_e2e_results.json")

asyncio.run(main())
