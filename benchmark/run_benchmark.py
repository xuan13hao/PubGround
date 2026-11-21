"""
Benchmark: claim -> evidence retrieval on SciFact (BEIR), no OpenAI key needed.

Compares three systems on the standard SciFact test split (300 claims, 5,183
abstracts), the de-facto benchmark for biomedical claim grounding:

  1. BM25                  — lexical baseline
  2. MedCPT dense          — MedCPT bi-encoder (query + article encoders)
  3. BM25 -> MedCPT rerank — BM25 top-100 reranked by the MedCPT cross-encoder
                             (the reranker this project adds, Stage 4.5)

Metrics (BEIR-standard + extras), macro-averaged over queries:
  nDCG@10, Recall@10, Recall@100, MRR@10, P@1

Published anchors (for sanity): BM25 nDCG@10 ~0.665; MedCPT dense ~0.761
(Jin et al. 2023; BEIR leaderboard).

Usage:
  python benchmark/run_benchmark.py [--max-queries N] [--skip-dense] [--topk 100]
"""
import argparse, json, math, re, sys, time
from collections import defaultdict

import pandas as pd

DATA = __file__.rsplit("/", 1)[0] + "/data"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ----------------------------- data -----------------------------
def load_data():
    corpus = pd.read_parquet(f"{DATA}/corpus.parquet")
    queries = pd.read_parquet(f"{DATA}/queries.parquet")
    qrels_df = pd.read_csv(f"{DATA}/qrels_test.tsv", sep="\t")

    docs = {}  # id -> (title, text)
    for _, r in corpus.iterrows():
        docs[str(r["_id"])] = (str(r["title"] or ""), str(r["text"] or ""))

    qrels = defaultdict(set)  # qid -> {gold doc ids}
    for _, r in qrels_df.iterrows():
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].add(str(r["corpus-id"]))

    test_q = {}  # qid -> claim text
    for _, r in queries.iterrows():
        qid = str(r["_id"])
        if qid in qrels:
            claim = str(r["text"] or "").strip() or str(r["title"] or "").strip()
            test_q[qid] = claim
    return docs, test_q, qrels


# ----------------------------- metrics -----------------------------
def metrics_for(ranked_ids, gold, ks=(10, 100)):
    """ranked_ids: list of doc ids best-first. gold: set. Returns dict."""
    out = {}
    # nDCG@10
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:10]):
        if d in gold:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0
    # Recall@k
    for k in ks:
        hit = len(gold & set(ranked_ids[:k]))
        out[f"recall@{k}"] = hit / len(gold) if gold else 0.0
    # MRR@10
    mrr = 0.0
    for i, d in enumerate(ranked_ids[:10]):
        if d in gold:
            mrr = 1.0 / (i + 1)
            break
    out["mrr@10"] = mrr
    # P@1
    out["p@1"] = 1.0 if ranked_ids and ranked_ids[0] in gold else 0.0
    return out


def aggregate(per_query):
    keys = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "p@1"]
    n = len(per_query)
    return {k: round(sum(q[k] for q in per_query) / n, 4) for k in keys}


# ----------------------------- BM25 -----------------------------
_tok = re.compile(r"[A-Za-z0-9]+")
def tok(s):
    return [w.lower() for w in _tok.findall(s)]


def run_bm25(docs, test_q, topk):
    from rank_bm25 import BM25Okapi
    doc_ids = list(docs.keys())
    log(f"[bm25] tokenizing {len(doc_ids)} docs ...")
    corpus_tok = [tok(docs[d][0] + " " + docs[d][1]) for d in doc_ids]
    bm25 = BM25Okapi(corpus_tok)
    runs = {}
    log(f"[bm25] scoring {len(test_q)} queries ...")
    import numpy as np
    for qid, claim in test_q.items():
        scores = bm25.get_scores(tok(claim))
        idx = np.argsort(scores)[::-1][:topk]
        runs[qid] = [(doc_ids[i], float(scores[i])) for i in idx]
    return runs


# ----------------------------- MedCPT encoders -----------------------------
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _encode(model, tok_, inputs, max_len, bs, dev):
    import torch
    embs = []
    for i in range(0, len(inputs), bs):
        batch = inputs[i : i + bs]
        enc = tok_(batch, truncation=True, padding=True, max_length=max_len,
                   return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**enc).last_hidden_state[:, 0, :]  # CLS
        embs.append(out.cpu())
        if (i // bs) % 20 == 0:
            log(f"    encoded {i+len(batch)}/{len(inputs)}")
    return torch.cat(embs, dim=0)


def run_medcpt_dense(docs, test_q, topk, bs=64):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = _device()
    doc_ids = list(docs.keys())

    log("[dense] loading MedCPT article/query encoders ...")
    a_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    a_mod = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(dev).eval()
    q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    q_mod = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(dev).eval()

    log(f"[dense] encoding {len(doc_ids)} articles ...")
    arts = [[docs[d][0], docs[d][1]] for d in doc_ids]  # [title, abstract] pairs
    doc_emb = _encode(a_mod, a_tok, arts, 512, bs, dev)

    log(f"[dense] encoding {len(test_q)} queries ...")
    qids = list(test_q.keys())
    q_emb = _encode(q_mod, q_tok, [test_q[q] for q in qids], 64, bs, dev)

    log("[dense] scoring (dot product) ...")
    scores = q_emb @ doc_emb.T  # (nq, ndoc)
    runs = {}
    vals, idx = torch.topk(scores, k=min(topk, len(doc_ids)), dim=1)
    for r, qid in enumerate(qids):
        runs[qid] = [(doc_ids[int(j)], float(v)) for j, v in zip(idx[r], vals[r])]
    del a_mod, q_mod
    torch.cuda.empty_cache() if dev == "cuda" else None
    return runs


def rerank_crossencoder(docs, test_q, bm25_runs, topk, bs=32):
    """Rerank BM25 top-k with the MedCPT cross-encoder (this project's reranker)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = _device()
    log("[rerank] loading MedCPT cross-encoder ...")
    tok_ = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    mod = AutoModelForSequenceClassification.from_pretrained(
        "ncbi/MedCPT-Cross-Encoder").to(dev).eval()

    runs = {}
    qids = list(test_q.keys())
    for n, qid in enumerate(qids):
        cand = [d for d, _ in bm25_runs[qid][:topk]]
        claim = test_q[qid]
        pairs = [[claim, (docs[d][0] + " " + docs[d][1])[:2000]] for d in cand]
        scores = []
        for i in range(0, len(pairs), bs):
            enc = tok_(pairs[i:i+bs], truncation=True, padding=True,
                       max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                lo = mod(**enc).logits.squeeze(-1)
            scores.extend(lo.cpu().tolist() if lo.dim() else [float(lo)])
        order = sorted(zip(cand, scores), key=lambda x: -x[1])
        runs[qid] = order
        if n % 50 == 0:
            log(f"    reranked {n}/{len(qids)}")
    return runs


# ----------------------------- driver -----------------------------
def evaluate(runs, qrels):
    per = [metrics_for([d for d, _ in runs[q]], qrels[q]) for q in runs]
    return aggregate(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--skip-dense", action="store_true")
    args = ap.parse_args()

    docs, test_q, qrels = load_data()
    if args.max_queries:
        test_q = dict(list(test_q.items())[: args.max_queries])
        qrels = {q: qrels[q] for q in test_q}
    log(f"corpus={len(docs)} test_queries={len(test_q)} topk={args.topk}")

    results, timing = {}, {}

    t = time.time()
    bm25 = run_bm25(docs, test_q, args.topk)
    results["BM25"] = evaluate(bm25, qrels); timing["BM25"] = round(time.time()-t, 1)
    log("BM25:", results["BM25"])

    t = time.time()
    rr = rerank_crossencoder(docs, test_q, bm25, args.topk)
    results["BM25 + MedCPT rerank"] = evaluate(rr, qrels)
    timing["BM25 + MedCPT rerank"] = round(time.time()-t, 1)
    log("Rerank:", results["BM25 + MedCPT rerank"])

    if not args.skip_dense:
        t = time.time()
        dense = run_medcpt_dense(docs, test_q, args.topk)
        results["MedCPT dense"] = evaluate(dense, qrels)
        timing["MedCPT dense"] = round(time.time()-t, 1)
        log("Dense:", results["MedCPT dense"])

    out = {
        "dataset": "SciFact (BEIR)",
        "corpus_size": len(docs),
        "n_queries": len(test_q),
        "topk": args.topk,
        "metrics": ["ndcg@10", "recall@10", "recall@100", "mrr@10", "p@1"],
        "results": results,
        "timing_sec": timing,
        "published_anchors": {"BM25 ndcg@10": 0.665, "MedCPT dense ndcg@10": 0.761},
    }
    with open(f"{DATA}/../results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
