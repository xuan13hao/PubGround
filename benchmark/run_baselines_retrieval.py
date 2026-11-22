"""
Layer 1 comparison: this project's retriever / reranker (BM25, MedCPT) vs. widely used
open-source retrieval models, on the exact same SciFact (BEIR) setup.

First-stage retrievers (full 5,183-abstract corpus):
  BM25 (rank_bm25) | MedCPT dense (ours) | e5-large-v2 | bge-large-en-v1.5 |
  SPLADE (cocondenser-ensembledistil) | Qwen3-Embedding-4B
Rerankers (identical candidate lists: BM25 top-100 = standard BEIR protocol, and
Hybrid RRF(BM25, MedCPT dense) top-100 = this project's best first stage):
  MedCPT cross-encoder (ours) | ms-marco-MiniLM-L-12-v2 | bge-reranker-v2-m3 |
  monoT5-3B (msmarco-10k) | Qwen3-Reranker-4B

Every run is cached under data/baseline_runs/, so a job that hits its time limit
can simply be resubmitted.

Usage:
  python run_baselines_retrieval.py --only dense      # first-stage retrievers
  python run_baselines_retrieval.py --only rerank     # rerankers
  python run_baselines_retrieval.py                   # everything, then write results
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_benchmark import load_data, tok, log  # noqa: E402
from run_llm_benchmark import rrf, evaluate  # noqa: E402

DATA = os.environ.get("PUBGROUND_BENCH_DATA") or os.path.join(HERE, "data")  # override with PUBGROUND_BENCH_DATA
RUNS = f"{DATA}/baseline_runs"
TOPK = 100
QWEN_TASK = "Given a scientific claim, retrieve documents that support or refute the claim"


def doc_text(docs, d):
    return (docs[d][0] + " " + docs[d][1]).strip()


def cached(name):
    p = f"{RUNS}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def save(name, runs):
    os.makedirs(RUNS, exist_ok=True)
    json.dump(runs, open(f"{RUNS}/{name}.json", "w"))


def dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def topk_runs(q_emb, d_emb, qids, doc_ids, k=TOPK):
    import torch
    s = q_emb.float() @ d_emb.float().T
    v, i = torch.topk(s, k=min(k, len(doc_ids)), dim=1)
    return {q: [(doc_ids[int(j)], float(x)) for j, x in zip(i[r], v[r])] for r, q in enumerate(qids)}


# ----------------------------- first-stage retrievers -----------------------------
def encode_hf(name, texts, pool, max_len, bs=64, normalize=True, left_pad=False):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    t = AutoTokenizer.from_pretrained(name, padding_side="left" if left_pad else "right")
    m = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16).to(dev()).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = t(texts[i:i + bs], truncation=True, padding=True, max_length=max_len,
                return_tensors="pt").to(dev())
        with torch.no_grad():
            h = m(**enc).last_hidden_state
        if pool == "cls":
            e = h[:, 0]
        elif pool == "mean":
            msk = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * msk).sum(1) / msk.sum(1)
        elif pool == "last":  # left-padded -> last position is the final token
            e = h[:, -1]
        e = e.float()
        out.append(F.normalize(e, dim=-1) if normalize else e)
    del m
    torch.cuda.empty_cache()
    return torch.cat(out)


def run_dense(name, docs, test_q):
    doc_ids, qids = list(docs), list(test_q)
    D = [doc_text(docs, d) for d in doc_ids]
    Q = [test_q[q] for q in qids]
    if name == "e5-large-v2":
        de = encode_hf("intfloat/e5-large-v2", ["passage: " + x for x in D], "mean", 512)
        qe = encode_hf("intfloat/e5-large-v2", ["query: " + x for x in Q], "mean", 512)
    elif name == "bge-large-en-v1.5":
        de = encode_hf("BAAI/bge-large-en-v1.5", D, "cls", 512)
        qe = encode_hf("BAAI/bge-large-en-v1.5",
                       ["Represent this sentence for searching relevant passages: " + x for x in Q], "cls", 512)
    elif name == "Qwen3-Embedding-4B":
        de = encode_hf("Qwen/Qwen3-Embedding-4B", D, "last", 512, bs=16, left_pad=True)
        qe = encode_hf("Qwen/Qwen3-Embedding-4B", [f"Instruct: {QWEN_TASK}\nQuery:{x}" for x in Q],
                       "last", 512, bs=16, left_pad=True)
    elif name == "SPLADE":
        de, qe = splade_encode(D), splade_encode(Q)
    else:
        raise ValueError(name)
    return topk_runs(qe, de, qids, doc_ids)


def splade_encode(texts, bs=32):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    n = "naver/splade-cocondenser-ensembledistil"
    t = AutoTokenizer.from_pretrained(n)
    m = AutoModelForMaskedLM.from_pretrained(n).to(dev()).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = t(texts[i:i + bs], truncation=True, padding=True, max_length=512, return_tensors="pt").to(dev())
        with torch.no_grad():
            lo = m(**enc).logits
        v = torch.max(torch.log1p(torch.relu(lo)) * enc["attention_mask"].unsqueeze(-1), dim=1).values
        out.append(v.half())
    del m
    torch.cuda.empty_cache()
    return torch.cat(out)


# ----------------------------- rerankers -----------------------------
class PairScorer:
    """score(pairs) -> list[float] for [(query, doc_text)]; higher = more relevant."""

    def __init__(self, name):
        import torch
        self.name = name
        self.d = dev()
        if name in ("MedCPT", "MiniLM-L12", "bge-reranker-v2-m3", "OpenScholar-Reranker", "bge-reranker-large"):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            hf = {"MedCPT": "ncbi/MedCPT-Cross-Encoder",
                  "MiniLM-L12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                  "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
                  "OpenScholar-Reranker": "OpenSciLM/OpenScholar_Reranker",
                  "bge-reranker-large": "BAAI/bge-reranker-large"}[name]  # OpenScholar reranker's base model
            # OpenScholar ships only a sentencepiece model: use the slow tokenizer (no protobuf conversion)
            self.t = AutoTokenizer.from_pretrained(hf, use_fast=name != "OpenScholar-Reranker")
            self.m = AutoModelForSequenceClassification.from_pretrained(
                hf, torch_dtype=torch.float16).to(self.d).eval()
            self.kind, self.bs = "ce", 64
        elif name == "monoT5-3B":
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            hf = "castorini/monot5-3b-msmarco-10k"
            # slow (sentencepiece) tokenizer: the fast conversion needs a protobuf newer than the cluster's
            self.t = AutoTokenizer.from_pretrained(hf, use_fast=False, legacy=True)
            self.m = T5ForConditionalGeneration.from_pretrained(hf, torch_dtype=torch.bfloat16).to(self.d).eval()
            self.true_id = self.t.encode("true", add_special_tokens=False)[0]
            self.false_id = self.t.encode("false", add_special_tokens=False)[0]
            self.kind, self.bs = "t5", 32
        elif name == "Qwen3-Reranker-4B":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            hf = "Qwen/Qwen3-Reranker-4B"
            self.t = AutoTokenizer.from_pretrained(hf, padding_side="left")
            self.m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16).to(self.d).eval()
            self.yes = self.t.convert_tokens_to_ids("yes")
            self.no = self.t.convert_tokens_to_ids("no")
            self.prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
                           "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                           "<|im_end|>\n<|im_start|>user\n")
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.kind, self.bs = "qwen", 16
        else:
            raise ValueError(name)

    def score(self, pairs):
        import torch
        out = []
        for i in range(0, len(pairs), self.bs):
            b = pairs[i:i + self.bs]
            with torch.no_grad():
                if self.kind == "ce":
                    enc = self.t([q for q, _ in b], [d for _, d in b], truncation="only_second",
                                 padding=True, max_length=512, return_tensors="pt").to(self.d)
                    lo = self.m(**enc).logits
                    out += lo[:, 0].float().cpu().tolist() if lo.shape[-1] == 1 else \
                        (lo[:, 1] - lo[:, 0]).float().cpu().tolist()
                elif self.kind == "t5":
                    txt = [f"Query: {q} Document: {d} Relevant:" for q, d in b]
                    enc = self.t(txt, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.d)
                    dec = torch.full((len(b), 1), self.m.config.decoder_start_token_id, device=self.d)
                    lo = self.m(**enc, decoder_input_ids=dec).logits[:, 0]
                    out += (lo[:, self.true_id] - lo[:, self.false_id]).float().cpu().tolist()
                else:
                    body = [f"<Instruct>: {QWEN_TASK}\n<Query>: {q}\n<Document>: {d}" for q, d in b]
                    # truncate the document part so the whole prompt stays within 1024 tokens
                    body = [self.t.decode(self.t.encode(x, add_special_tokens=False)[:960]) for x in body]
                    enc = self.t([self.prefix + x + self.suffix for x in body], padding=True,
                                 return_tensors="pt").to(self.d)
                    lo = self.m(**enc).logits[:, -1]
                    two = torch.stack([lo[:, self.no], lo[:, self.yes]], dim=1).float()
                    out += torch.log_softmax(two, dim=1)[:, 1].cpu().tolist()
            if (i // self.bs) % 100 == 0:
                log(f"    [{self.name}] {i + len(b)}/{len(pairs)}")
        return out

    def close(self):
        import torch
        del self.m
        torch.cuda.empty_cache()


def first_stages(docs, test_q):
    """BM25 and Hybrid(BM25, MedCPT dense) top-100 candidate lists, shared by all rerankers."""
    bm = cached("BM25")
    if bm is None:
        from rank_bm25 import BM25Okapi
        import numpy as np
        ids = list(docs)
        idx = BM25Okapi([tok(doc_text(docs, d)) for d in ids])
        bm = {}
        for q in test_q:
            s = idx.get_scores(tok(test_q[q]))
            o = np.argsort(s)[::-1][:TOPK]
            bm[q] = [(ids[i], float(s[i])) for i in o]
        save("BM25", bm)
    dense = cached("MedCPT dense")
    if dense is None:
        dense = json.load(open(f"{DATA}/dense_runs_top{TOPK}.json"))  # written by run_llm_benchmark.py
        save("MedCPT dense", dense)
    hyb = {q: rrf([bm[q], dense[q]], TOPK) for q in test_q}
    return {"BM25": bm, "Hybrid": hyb}


DENSE = ["e5-large-v2", "bge-large-en-v1.5", "SPLADE", "Qwen3-Embedding-4B"]
RERANK = ["MedCPT", "MiniLM-L12", "bge-reranker-v2-m3", "monoT5-3B", "Qwen3-Reranker-4B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dense", "rerank", "report"], default=None)
    ap.add_argument("--models", default="", help="comma list to restrict models")
    args = ap.parse_args()
    restrict = set(m for m in args.models.split(",") if m)

    docs, test_q, qrels = load_data()
    fs = first_stages(docs, test_q)
    timing_path = f"{RUNS}/timing.json"
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}

    if args.only in (None, "dense"):
        for n in DENSE:
            if (restrict and n not in restrict) or cached(n):
                continue
            log(f"[dense] {n}")
            t = time.time()
            save(n, run_dense(n, docs, test_q))
            timing[n] = round(time.time() - t, 1)
            json.dump(timing, open(timing_path, "w"), indent=1)

    if args.only in (None, "rerank"):
        for n in RERANK:
            if restrict and n not in restrict:
                continue
            todo = [c for c in fs if not cached(f"{c} + {n}")]
            if not todo:
                continue
            log(f"[rerank] {n} on {todo}")
            t = time.time()
            sc = PairScorer(n)
            # score the union of candidate pairs once
            pairs = sorted({(q, d) for c in todo for q in test_q for d, _ in fs[c][q]})
            scores = dict(zip(pairs, sc.score([(test_q[q], doc_text(docs, d)[:4000]) for q, d in pairs])))
            sc.close()
            for c in todo:
                save(f"{c} + {n}", {q: sorted(((d, scores[(q, d)]) for d, _ in fs[c][q]),
                                              key=lambda x: -x[1]) for q in test_q})
            timing[f"rerank {n}"] = round(time.time() - t, 1)
            json.dump(timing, open(timing_path, "w"), indent=1)

    # ---- report: everything that has a cached run ----
    rows = {}
    for n in ["BM25", "MedCPT dense"] + DENSE:
        r = cached(n)
        if r:
            rows[n] = evaluate(r, qrels)
    for c in ("BM25", "Hybrid"):
        if c == "Hybrid":
            rows["Hybrid RRF (BM25 + MedCPT dense)"] = evaluate(fs["Hybrid"], qrels)
        for n in RERANK:
            r = cached(f"{c} + {n}")
            if r:
                rows[f"{c} top-100 -> {n}"] = evaluate(r, qrels)
    out = {"dataset": "SciFact (BEIR test)", "n_queries": len(test_q), "topk": TOPK,
           "results": rows, "timing_sec": timing,
           "ours": ["BM25", "MedCPT dense", "MedCPT"]}
    json.dump(out, open(f"{HERE}/results_baselines_retrieval.json", "w"), indent=2)
    for k, v in rows.items():
        print(f"{k:45s} nDCG@10={v['ndcg@10']:.4f} R@10={v['recall@10']:.4f} "
              f"R@100={v['recall@100']:.4f} MRR@10={v['mrr@10']:.4f} P@1={v['p@1']:.4f}")


if __name__ == "__main__":
    main()
