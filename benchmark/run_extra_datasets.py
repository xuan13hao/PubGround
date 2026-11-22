"""
Benchmark on two more datasets, same systems and metrics as the SciFact runs.

  scifact_open  SciFact-Open (Wadden et al. 2022): the 279 SciFact TEST claims retrieved against
                a 500K-abstract S2ORC corpus; 206 claims have evidence (460 evidence abstracts,
                SUPPORT 249 / CONTRADICT 211, from citations + pooling).  -> retrieval + verdict
  healthver     HealthVer test split (Sarrouti et al. 2021, MultiVerS-processed): 230 COVID health
                claims, each with 1-9 candidate abstracts labelled SUPPORT/CONTRADICT/NEI.
                -> verdict only (its 322-abstract corpus is too small for a retrieval benchmark)

Fair supervised baselines exist here (checked: no train/eval claim overlap):
  MultiVerS `scifact` checkpoint was trained on SciFact train+dev, never on the test claims;
  MultiVerS `healthver` checkpoint was trained on HealthVer train only.

Tasks (each resumable; caches live in data/extra/<dataset>/):
  --task encode   dense-encode the corpus in 50K-doc shards   (--models MedCPT,e5-large-v2,...)
  --task rerank   BM25, dense search, Hybrid RRF, rerankers on Hybrid top-100 (+ BM25 -> MedCPT)
  --task llm      Stage 1 LLM-QE -> BM25 -> MedCPT rerank, per --models LLM
  --task verdict  every verifier on the shared inputs
  --task report   write results_<dataset>.json
"""
import argparse, glob, json, math, os, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_benchmark import metrics_for, aggregate, tok, log  # noqa: E402
from run_llm_benchmark import (rrf, run_stage1,  # noqa: E402
                               run_verdict, scifact_abstract_metrics, three_class, REASONING_MANDATORY)
from llm_client import make_llm  # noqa: E402

DATA = os.path.join(HERE, "data")
TOPK = 100
QWEN_TASK = "Given a scientific claim, retrieve documents that support or refute the claim"


# ----------------------------- data -----------------------------
def load_dataset(name):
    """-> corpus {int doc_id: {"title", "abstract": [sentences]}},
          claims {str cid: {"claim", "evidence": {str doc: [{"label","sentences"}]}, "cand": [int doc]}}"""
    L = lambda p: [json.loads(l) for l in open(p)]
    if name == "scifact_open":
        base = f"{DATA}/scifact_open/data"
        corpus = {r["doc_id"]: {"title": r["title"], "abstract": r["abstract"]} for r in L(f"{base}/corpus.jsonl")}
        claims = {}
        for r in L(f"{base}/claims.jsonl"):
            ev = {d: [{"label": e["label"], "sentences": e["sentences"]}] for d, e in r["evidence"].items()}
            claims[str(r["id"])] = {"claim": r["claim"], "evidence": ev, "cand": [int(d) for d in r["evidence"]]}
    elif name == "healthver":
        base = f"{DATA}/ext_multivers/data/data/healthver"
        corpus = {r["doc_id"]: {"title": r["title"], "abstract": r["abstract"]} for r in L(f"{base}/corpus.jsonl")}
        claims = {str(r["id"]): {"claim": r["claim"].strip(), "evidence": r["evidence"], "cand": r["doc_ids"]}
                  for r in L(f"{base}/claims_test.jsonl")}
    else:
        raise ValueError(name)
    return corpus, claims


def out_dir(name):
    p = f"{DATA}/extra/{name}"
    os.makedirs(p, exist_ok=True)
    return p


def doc_text(corpus, d):
    return (corpus[d]["title"] + " " + " ".join(corpus[d]["abstract"])).strip()


def evaluate(runs, qrels):
    return aggregate([metrics_for([str(d) for d, _ in runs[q]], qrels[q]) for q in qrels])


# ----------------------------- BM25 (Lucene variant, scipy sparse) -----------------------------
class SparseBM25:
    """Lucene BM25 (k1=0.9, b=0.4, idf=log(1+(N-df+.5)/(df+.5))), the BEIR/Anserini setting."""

    def __init__(self, texts, k1=0.9, b=0.4):
        import numpy as np
        from scipy.sparse import csr_matrix
        vocab, rows, cols, vals = {}, [], [], []
        for i, t in enumerate(texts):
            cnt = defaultdict(int)
            for w in tok(t):
                cnt[w] += 1
            for w, c in cnt.items():
                rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); vals.append(c)
        tf = csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
        dl = np.asarray(tf.sum(1)).ravel()
        df = np.bincount(tf.indices, minlength=len(vocab))
        idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.W = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).tocsc()
        self.vocab = vocab

    def search(self, tokens, k):
        import numpy as np
        ids = [self.vocab[w] for w in tokens if w in self.vocab]
        if not ids:
            return []
        cnt = defaultdict(int)
        for i in ids:
            cnt[i] += 1
        s = np.zeros(self.W.shape[0], dtype=np.float32)
        for i, c in cnt.items():
            col = self.W.getcol(i)
            s[col.indices] += c * col.data
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(int(j), float(s[j])) for j in top if s[j] > 0]


# ----------------------------- dense encoding (sharded, resumable) -----------------------------
DENSE = {
    # name: (doc model, query model, pooling, doc prefix, query prefix, batch)
    "MedCPT dense": ("ncbi/MedCPT-Article-Encoder", "ncbi/MedCPT-Query-Encoder", "cls", None, None, 128),
    "e5-large-v2": ("intfloat/e5-large-v2", "intfloat/e5-large-v2", "mean", "passage: ", "query: ", 128),
    "bge-large-en-v1.5": ("BAAI/bge-large-en-v1.5", "BAAI/bge-large-en-v1.5", "cls", "",
                          "Represent this sentence for searching relevant passages: ", 128),
    "Qwen3-Embedding-4B": ("Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-4B", "last", "",
                           f"Instruct: {QWEN_TASK}\nQuery:", 32),
}
SHARD = 50000


def _encoder(hf, left_pad):
    import torch
    from transformers import AutoTokenizer, AutoModel
    t = AutoTokenizer.from_pretrained(hf, padding_side="left" if left_pad else "right")
    m = AutoModel.from_pretrained(hf, torch_dtype=torch.float16 if "MedCPT" in hf else torch.bfloat16)
    return t, m.to("cuda").eval()


def _embed(t, m, inputs, pool, bs, normalize, pairs=False):
    import torch
    import torch.nn.functional as F
    order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i][1] if pairs else inputs[i]))
    out = [None] * len(inputs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [inputs[i] for i in idx]
        enc = t([b[0] for b in batch], [b[1] for b in batch], truncation=True, padding=True, max_length=512,
                return_tensors="pt") if pairs else \
            t(batch, truncation=True, padding=True, max_length=512, return_tensors="pt")
        enc = enc.to("cuda")
        with torch.no_grad():
            h = m(**enc).last_hidden_state
        if pool == "cls":
            e = h[:, 0]
        elif pool == "mean":
            msk = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * msk).sum(1) / msk.sum(1)
        else:
            e = h[:, -1]
        e = e.float()
        e = F.normalize(e, dim=-1) if normalize else e
        for j, i in enumerate(idx):
            out[i] = e[j].half().cpu()
    return torch.stack(out)


def encode_corpus(name, model, corpus, deadline):
    import torch
    dm, qm, pool, dpre, _, bs = DENSE[model]
    d = f"{out_dir(name)}/emb/{model}"
    os.makedirs(d, exist_ok=True)
    ids = sorted(corpus)
    n_shards = math.ceil(len(ids) / SHARD)
    t = m = None
    for s in range(n_shards):
        p = f"{d}/shard_{s:03d}.pt"
        if os.path.exists(p):
            continue
        if time.time() > deadline:
            log(f"[encode {model}] stopping before shard {s} (time budget); resubmit to continue")
            return False
        if t is None:
            t, m = _encoder(dm, pool == "last")
        chunk = ids[s * SHARD:(s + 1) * SHARD]
        if model == "MedCPT dense":
            inp = [[corpus[i]["title"], " ".join(corpus[i]["abstract"])] for i in chunk]
            e = _embed(t, m, inp, pool, bs, normalize=False, pairs=True)
        else:
            inp = [dpre + doc_text(corpus, i) for i in chunk]
            e = _embed(t, m, inp, pool, bs, normalize=True)
        torch.save({"ids": chunk, "emb": e}, p + ".tmp")
        os.replace(p + ".tmp", p)
        log(f"[encode {model}] shard {s + 1}/{n_shards} done")
    if m is not None:
        del m
        torch.cuda.empty_cache()
    return True


def dense_search(name, model, queries, k=TOPK):
    import torch
    dm, qm, pool, _, qpre, _ = DENSE[model]
    shards = sorted(glob.glob(f"{out_dir(name)}/emb/{model}/shard_*.pt"))
    t, m = _encoder(qm, pool == "last")
    qids = list(queries)
    qin = [queries[q] if model == "MedCPT dense" else qpre + queries[q] for q in qids]
    qe = _embed(t, m, qin, pool, 32, normalize=model != "MedCPT dense").float().cuda()
    del m
    best = [[] for _ in qids]
    for p in shards:
        sh = torch.load(p)
        sc = qe @ sh["emb"].float().cuda().T
        v, i = torch.topk(sc, k=min(k, sc.shape[1]), dim=1)
        for r in range(len(qids)):
            best[r] += [(sh["ids"][int(j)], float(x)) for j, x in zip(i[r], v[r])]
    torch.cuda.empty_cache()
    return {q: sorted(best[r], key=lambda x: -x[1])[:k] for r, q in enumerate(qids)}


# ----------------------------- run cache -----------------------------
def run_path(name, sys_name):
    return f"{out_dir(name)}/runs/{sys_name}.json"


def load_run(name, sys_name):
    p = run_path(name, sys_name)
    return {q: [(int(d), s) for d, s in v] for q, v in json.load(open(p)).items()} if os.path.exists(p) else None


def save_run(name, sys_name, runs):
    os.makedirs(f"{out_dir(name)}/runs", exist_ok=True)
    json.dump(runs, open(run_path(name, sys_name), "w"))


class PairCache:
    """Reranker scores keyed by (reranker, claim, doc); persisted per dataset."""

    def __init__(self, name, reranker):
        self.p = f"{out_dir(name)}/rerank_{reranker}.json"
        self.c = json.load(open(self.p)) if os.path.exists(self.p) else {}
        self.reranker = reranker

    def score(self, claims, corpus, cand):
        from run_baselines_retrieval import PairScorer
        todo = sorted({(q, d) for q, ds in cand.items() for d in ds if f"{q}|{d}" not in self.c})
        if todo:
            log(f"[rerank {self.reranker}] {len(todo)} new pairs")
            sc = PairScorer(self.reranker)
            vals = sc.score([(claims[q]["claim"], doc_text(corpus, d)[:4000]) for q, d in todo])
            sc.close()
            for (q, d), v in zip(todo, vals):
                self.c[f"{q}|{d}"] = v
            json.dump(self.c, open(self.p, "w"))
        return {q: sorted(((d, self.c[f"{q}|{d}"]) for d in ds), key=lambda x: -x[1]) for q, ds in cand.items()}


def first_stage(name, corpus, claims, need_bm25_index=False):
    """BM25, MedCPT dense, Hybrid runs (cached). Returns (bm25, dense, hybrid, bm25_index|None)."""
    idx = None
    bm = load_run(name, "BM25")
    if bm is None or need_bm25_index:
        t = time.time()
        ids = sorted(corpus)
        idx_obj = SparseBM25([doc_text(corpus, i) for i in ids])
        idx = lambda toks, k: [(ids[j], s) for j, s in idx_obj.search(toks, k)]
        log(f"[bm25] indexed {len(ids)} docs in {time.time() - t:.0f}s")
        if bm is None:
            bm = {q: idx(tok(c["claim"]), TOPK) for q, c in claims.items()}
            save_run(name, "BM25", bm)
    dense = load_run(name, "MedCPT dense")
    if dense is None:
        dense = dense_search(name, "MedCPT dense", {q: c["claim"] for q, c in claims.items()})
        save_run(name, "MedCPT dense", dense)
    hyb = {q: rrf([bm[q], dense[q]], TOPK) for q in claims}
    return bm, dense, hyb, idx


RERANKERS = ["MedCPT", "MiniLM-L12", "bge-reranker-v2-m3", "monoT5-3B", "Qwen3-Reranker-4B"]


# ----------------------------- verdict -----------------------------
def verdict(name, corpus, claims, llms, skip):
    from run_baselines_verdict import VeriSci, MultiVerS
    sci = corpus  # same {"title","abstract"} shape as the SciFact corpus
    if name == "healthver":
        oracle_pairs = [(c, d) for c, cl in claims.items() for d in cl["cand"]]
        open_pairs = []
    else:
        oracle_pairs = [(c, int(d)) for c, cl in claims.items() for d in cl["evidence"]]
        rr = load_run(name, "Hybrid + MedCPT")
        assert rr, "run --task rerank first"
        open_pairs = [(c, d) for c in claims for d, _ in rr[c][:3]]
    gold = [(claims[c]["evidence"].get(str(d)) or [{"label": "NEI"}])[0]["label"] for c, d in oracle_pairs]
    pairs = sorted(set(oracle_pairs) | set(open_pairs))
    log(f"[verdict {name}] oracle={len(oracle_pairs)} open={len(open_pairs)} union={len(pairs)}")
    store_p = f"{out_dir(name)}/verdict_preds.json"
    store = json.load(open(store_p)) if os.path.exists(store_p) else {}

    neural = [("VeriSci", VeriSci, ()),
              ("MultiVerS-zs (fever_sci)", MultiVerS, ("fever_sci",)),
              ("MultiVerS-scifact (supervised, SciFact train+dev)", MultiVerS, ("scifact",))]
    if name == "healthver":
        neural.append(("MultiVerS-healthver (supervised, HealthVer train)", MultiVerS, ("healthver",)))
    preds = {}
    for vname, cls, a in neural:
        if vname.split()[0] in skip:
            continue
        have = store.setdefault(vname, {})
        missing = [(c, d) for c, d in pairs if f"{c}|{d}" not in have]
        if missing:
            log(f"[{vname}] {len(missing)} new pairs")
            m = cls(*a)
            for c, d in missing:
                have[f"{c}|{d}"] = m.predict(claims[c]["claim"], sci[d]["title"], sci[d]["abstract"])
            if hasattr(m, "close"):
                m.close()
            del m
            json.dump(store, open(store_p, "w"))
        preds[vname] = {(c, d): tuple(store[vname][f"{c}|{d}"]) for c, d in pairs}
    for model in llms:
        llm = make_llm(model, reasoning=None if model in REASONING_MANDATORY else False)
        pv, n_fail = run_verdict(llm, claims, sci, pairs)
        preds["LLM " + model.split("/")[-1]] = pv
        log(f"[verdict LLM {model}] new calls={llm.n_calls} cost=${llm.cost:.3f} parse_fail={n_fail}")

    res = {}
    for k, pv in preds.items():
        orc, opn = defaultdict(dict), defaultdict(dict)
        for c, d in oracle_pairs:
            orc[c][d] = pv[(c, d)]
        for c, d in open_pairs:
            opn[c][d] = pv[(c, d)]
        r = {"oracle": three_class(gold, [pv[p][0] for p in oracle_pairs]),
             "oracle_abstract_level": scifact_abstract_metrics(claims, orc)}
        if open_pairs:
            r["open_abstract_level"] = scifact_abstract_metrics(claims, opn)
        res[k] = r
    json.dump(res, open(f"{out_dir(name)}/verdict_results.json", "w"), indent=1)
    if name == "healthver":  # verdict-only dataset: this is its whole result file
        json.dump({"dataset": "HealthVer test", "n_claims": len(claims), "n_pairs": len(oracle_pairs),
                   "verdict": res}, open(f"{HERE}/results_healthver.json", "w"), indent=2)
    return res


# ----------------------------- driver -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scifact_open", "healthver"])
    ap.add_argument("--task", required=True, choices=["encode", "rerank", "llm", "verdict", "report"])
    ap.add_argument("--models", default="")
    ap.add_argument("--budget-min", type=float, default=52, help="stop encoding before this many minutes")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    t0 = time.time()
    models = [m for m in args.models.split(",") if m]
    corpus, claims = load_dataset(args.dataset)
    qrels = {q: set(c["evidence"]) for q, c in claims.items() if c["evidence"]}
    log(f"{args.dataset}: corpus={len(corpus)} claims={len(claims)} with_evidence={len(qrels)}")

    if args.task == "encode":
        for m in models or list(DENSE):
            if not encode_corpus(args.dataset, m, corpus, t0 + args.budget_min * 60):
                sys.exit(3)

    elif args.task == "rerank":
        bm, dense, hyb, _ = first_stage(args.dataset, corpus, claims)
        for m in [x for x in DENSE if x != "MedCPT dense"]:
            if load_run(args.dataset, m) is None and \
                    len(glob.glob(f"{out_dir(args.dataset)}/emb/{m}/shard_*.pt")) == math.ceil(len(corpus) / SHARD):
                save_run(args.dataset, m, dense_search(args.dataset, m, {q: c["claim"] for q, c in claims.items()}))
        save_run(args.dataset, "Hybrid", hyb)
        for r in models or RERANKERS:
            if load_run(args.dataset, f"Hybrid + {r}") is None:
                save_run(args.dataset, f"Hybrid + {r}",
                         PairCache(args.dataset, r).score(claims, corpus, {q: [d for d, _ in hyb[q]] for q in claims}))
        if load_run(args.dataset, "BM25 + MedCPT") is None:
            save_run(args.dataset, "BM25 + MedCPT",
                     PairCache(args.dataset, "MedCPT").score(claims, corpus, {q: [d for d, _ in bm[q]] for q in claims}))

    elif args.task == "llm":
        bm, dense, hyb, idx = first_stage(args.dataset, corpus, claims, need_bm25_index=True)
        test_q = {q: c["claim"] for q, c in claims.items()}
        rcache = PairCache(args.dataset, "MedCPT")
        info = {}
        for model in models:
            tag = model.split("/")[-1]
            llm = make_llm(model, reasoning=None if model in REASONING_MANDATORY else False)
            # identical Stage 1 handling as the SciFact run (prompt, token budget, truncation salvage)
            qe, _, n_fb = run_stage1(llm, test_q, idx, TOPK)
            save_run(args.dataset, f"LLM-QE [{tag}]", qe)
            qe_rr = rcache.score(claims, corpus, {q: [d for d, _ in qe[q]] for q in claims})
            save_run(args.dataset, f"LLM-QE + MedCPT [{tag}]", qe_rr)
            info[tag] = {"stage1_fallbacks": n_fb, "new_calls": llm.n_calls, "cost": round(llm.cost, 4)}
            log(f"[llm {tag}] {info[tag]}")
        p = f"{out_dir(args.dataset)}/llm_info.json"
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.update(info)
        json.dump(old, open(p, "w"), indent=1)

    elif args.task == "verdict":
        res = verdict(args.dataset, corpus, claims, models, set(args.skip.split(",")))
        for k, v in res.items():
            line = f"{k:52s} oracle acc={v['oracle']['accuracy']:.3f} mF1={v['oracle']['macro_f1']:.3f}"
            line += f" | AL label F1={v['oracle_abstract_level']['label_only']['F1']:.3f}"
            if "open_abstract_level" in v:
                o = v["open_abstract_level"]["label_only"]
                line += f" | open label F1={o['F1']:.3f} (P {o['P']:.3f} R {o['R']:.3f})"
            print(line)

    if args.task in ("report", "rerank", "llm") and args.dataset == "scifact_open":
        rows = {}
        for p in sorted(glob.glob(f"{out_dir(args.dataset)}/runs/*.json")):
            n = os.path.basename(p)[:-5]
            rows[n] = evaluate(load_run(args.dataset, n), {q: {str(d) for d in v} for q, v in qrels.items()})
        out = {"dataset": args.dataset, "n_claims": len(claims), "n_eval_claims": len(qrels),
               "corpus_size": len(corpus), "retrieval": rows}
        vp = f"{out_dir(args.dataset)}/verdict_results.json"
        if os.path.exists(vp):
            out["verdict"] = json.load(open(vp))
        json.dump(out, open(f"{HERE}/results_{args.dataset}.json", "w"), indent=2)
        for k, v in rows.items():
            print(f"{k:55s} nDCG@10={v['ndcg@10']:.4f} R@10={v['recall@10']:.4f} R@100={v['recall@100']:.4f} "
                  f"MRR@10={v['mrr@10']:.4f} P@1={v['p@1']:.4f}")


if __name__ == "__main__":
    main()
