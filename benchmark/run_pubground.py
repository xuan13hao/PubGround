"""
Benchmark PubGround itself: every PubGround number comes from the package API
(``LocalCorpus``, ``PubMedSentenceMatcher.rank_abstracts``, ``VerdictJudge``); external systems are
scored on the same candidates and inputs.

  --dataset scifact | scifact_open | healthver
  --task index       build PubGround's MedCPT dense index and the OpenScholar retriever index
         pubground   PubGround rankings: hybrid (no rerank), hybrid + MedCPT, Stage 1 LLM-QE + hybrid + MedCPT
         rerankers   external rerankers on PubGround's hybrid top-100 (+ OpenScholar retriever -> reranker)
         timing      reranker throughput: model loaded, warm-up, 2,000 fixed pairs x 3 repeats
         verdict     PubGround judge (VerdictJudge) and external verifiers on gold / top-3 inputs
LLM calls go through benchmark/llm_client.py's cache (no new calls when cached).
Outputs: data/pubground/<dataset>/{runs/*.json, verdict_preds.json, timing.json}
"""
import argparse, asyncio, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from run_benchmark import log  # noqa: E402
from llm_client import make_llm  # noqa: E402
from run_llm_benchmark import REASONING_MANDATORY  # noqa: E402
from pubmed_sentence_matcher import PubMedSentenceMatcher  # noqa: E402
from pubmed_sentence_matcher.retrieval import LocalCorpus, MedCPTDenseIndex, doc_text, tokenize  # noqa: E402
from pubmed_sentence_matcher.verdict import VerdictJudge  # noqa: E402

DATA = os.environ.get("PUBGROUND_BENCH_DATA") or os.path.join(HERE, "data")  # override with PUBGROUND_BENCH_DATA
LLMS = {"claude-haiku-5.5": "anthropic/claude-haiku-5.5",
        "deepseek-v4.1-flash-1017": "deepseek/deepseek-v4.1-flash-1017",
        "claude-sonnet-5.5": "anthropic/claude-sonnet-5.5",
        "qwen3-8b": "local:/N/project/mrust01/LLM_NER/models/qwen3-8b"}
STAGE1_LLMS = ["claude-haiku-5.5", "deepseek-v4.1-flash-1017", "qwen3-8b"]
EXT_RERANKERS = ["MiniLM-L12", "bge-reranker-v2-m3", "monoT5-3B", "Qwen3-Reranker-4B", "OpenScholar-Reranker"]
TOPK = 100


# ----------------------------- data -----------------------------
def load(name):
    """-> corpus {str id: {title, abstract: [sentences]}}, claims {cid: {claim, evidence, gold_ids, cand}}"""
    L = lambda p: [json.loads(l) for l in open(p)]
    if name == "scifact":
        base = f"{DATA}/scifact_raw/data"
        corpus = {str(r["doc_id"]): {"title": r["title"], "abstract": r["abstract"]} for r in L(f"{base}/corpus.jsonl")}
        claims = {}
        for r in L(f"{base}/claims_dev.jsonl"):
            claims[str(r["id"])] = {"claim": r["claim"], "evidence": r["evidence"],
                                    # BEIR qrels = cited abstracts; dict.fromkeys drops the duplicate in claim 1245
                                    "cand": list(dict.fromkeys(str(d) for d in r["cited_doc_ids"]))}
        for c in claims.values():
            c["gold_ids"] = set(c["cand"])
        return corpus, claims
    from run_extra_datasets import load_dataset
    corpus, cl = load_dataset(name)
    corpus = {str(k): v for k, v in corpus.items()}
    claims = {}
    for c, x in cl.items():
        claims[c] = {"claim": x["claim"], "evidence": x["evidence"], "cand": [str(d) for d in x["cand"]],
                     "gold_ids": {str(d) for d in x["evidence"]}}
    return corpus, claims


def out(name, *parts):
    p = os.path.join(DATA, "pubground", name, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save_run(name, sysname, run):
    json.dump({q: [[d, float(s)] for d, s in v] for q, v in run.items()}, open(out(name, "runs", f"{sysname}.json"), "w"))


def load_run(name, sysname):
    p = out(name, "runs", f"{sysname}.json")
    return {q: [(str(d), s) for d, s in v] for q, v in json.load(open(p)).items()} if os.path.exists(p) else None


def cached_call(model_key):
    llm = make_llm(LLMS[model_key], reasoning=None if LLMS[model_key] in REASONING_MANDATORY else False)

    async def call(messages, temperature, max_tokens):
        return await asyncio.to_thread(llm.chat, messages, temperature, max_tokens, "pubground")
    return call, llm


def corpus_obj(name, corpus):
    idx = out(name, "index", "medcpt_dense.pt")
    if not os.path.exists(idx):
        MedCPTDenseIndex().build(corpus).save(idx)
    return LocalCorpus(corpus, bm25=True, dense=idx)


# ----------------------------- OpenScholar retriever -----------------------------
def openscholar_retrieve(name, corpus, claims):
    """Contriever-style mean pooling, dot product, full corpus."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    hf = "OpenSciLM/OpenScholar_Retriever"
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf, torch_dtype=torch.float16).cuda().eval()

    def enc(texts, bs=128):
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        res = [None] * len(texts)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            e = tok([texts[i] for i in idx], truncation=True, padding=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = model(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = ((h * m).sum(1) / m.sum(1)).float()
            for j, i in enumerate(idx):
                res[i] = v[j].half().cpu()
        return torch.stack(res)

    p = out(name, "index", "openscholar_retriever.pt")
    ids = list(corpus)
    if os.path.exists(p):
        emb = torch.load(p)["emb"]
    else:
        emb = enc([doc_text(corpus[d]) for d in ids])
        torch.save({"ids": ids, "emb": emb}, p)
    qids = list(claims)
    q = enc([claims[c]["claim"] for c in qids]).float().cuda()
    runs = {}
    for i in range(0, len(qids), 64):
        sc = q[i:i + 64] @ emb.float().cuda().T
        v, ix = torch.topk(sc, k=TOPK, dim=1)
        for r in range(sc.shape[0]):
            runs[qids[i + r]] = [(ids[int(j)], float(x)) for j, x in zip(ix[r], v[r])]
    return runs


# ----------------------------- tasks -----------------------------
def task_pubground(name, corpus, claims):
    lc = corpus_obj(name, corpus)
    m0 = PubMedSentenceMatcher(local_corpus=lc)
    rr = m0.reranker
    configs = [("PubGround BM25", None), ("PubGround MedCPT dense", None),
               ("PubGround hybrid", False), ("PubGround hybrid + MedCPT", False)]
    configs += [(f"PubGround Stage1 [{k}] + hybrid + MedCPT", k) for k in STAGE1_LLMS]
    for sysname, llm_key in configs:
        if load_run(name, sysname):
            continue
        t = time.time()
        if sysname == "PubGround BM25":
            run = {c: lc.bm25.search_tokens(tokenize(x["claim"]), TOPK) for c, x in claims.items()}
        elif sysname == "PubGround MedCPT dense":
            qids = list(claims)
            res = lc.dense.search_many([claims[c]["claim"] for c in qids], TOPK)
            run = dict(zip(qids, res))
        else:
            call = cached_call(llm_key)[0] if llm_key else None
            if not llm_key:
                async def call(*_):  # hybrid configs must never call an LLM
                    raise RuntimeError("no LLM in this configuration")
            m = PubMedSentenceMatcher(local_corpus=lc, llm_call=call)
            m.reranker = rr  # share the loaded cross-encoder
            rerank = sysname != "PubGround hybrid"

            async def go():
                return {c: await m.rank_abstracts(x["claim"], top_n=TOPK, expand_queries=bool(llm_key),
                                                  rerank=rerank) for c, x in claims.items()}
            run = asyncio.run(go())
        save_run(name, sysname, run)
        log(f"[{name}] {sysname}: {time.time() - t:.0f}s")


def task_rerankers(name, corpus, claims):
    from run_extra_datasets import PairCache
    hyb = load_run(name, "PubGround hybrid")
    assert hyb, "run --task pubground first"
    cl = {c: {"claim": x["claim"]} for c, x in claims.items()}
    cp = {int(d) if d.isdigit() else d: v for d, v in corpus.items()}
    cand = {c: [int(d) if d.isdigit() else d for d, _ in hyb[c]] for c in claims}
    for r in EXT_RERANKERS:
        sysname = f"PubGround hybrid -> {r}"
        if load_run(name, sysname):
            continue
        res = PairCache(name, r).score(cl, cp, cand)  # per-dataset pair-score cache, shared across runs
        save_run(name, sysname, {c: [(str(d), s) for d, s in v] for c, v in res.items()})
        log(f"[{name}] {sysname} done")
    if not load_run(name, "OpenScholar retriever"):
        save_run(name, "OpenScholar retriever", openscholar_retrieve(name, corpus, claims))
    if not load_run(name, "OpenScholar retriever -> OpenScholar-Reranker"):
        osr = load_run(name, "OpenScholar retriever")
        res = PairCache(name, "OpenScholar-Reranker").score(
            cl, cp, {c: [int(d) if d.isdigit() else d for d, _ in osr[c]] for c in claims})
        save_run(name, "OpenScholar retriever -> OpenScholar-Reranker",
                 {c: [(str(d), s) for d, s in v] for c, v in res.items()})


def task_timing(name, corpus, claims):
    """Throughput with the model already loaded: warm-up, then the same 2,000 pairs scored 3 times."""
    import statistics, torch
    from run_baselines_retrieval import PairScorer
    from pubmed_sentence_matcher.reranker import CrossEncoderReranker
    hyb = load_run(name, "PubGround hybrid")
    pairs = [(claims[c]["claim"], doc_text(corpus[d])) for c in sorted(claims) for d, _ in hyb[c]][:2000]
    res = {}
    for r in ["MedCPT (PubGround)"] + EXT_RERANKERS:
        if r == "MedCPT (PubGround)":
            sc = CrossEncoderReranker(device="cuda")   # the package's reranker, as PubGround calls it
            sc._ensure_loaded()

            def fn(P, sc=sc):  # one call per claim with all its candidates, as in rank_abstracts
                by = {}
                for q, d in P:
                    by.setdefault(q, []).append(d)
                return [x for q, ds in by.items() for x in sc._score_sync(q, ds)]
        else:
            ps = PairScorer(r)
            fn = ps.score
        fn(pairs[:64])  # warm-up
        ts = []
        for _ in range(3):
            torch.cuda.synchronize(); t = time.time(); fn(pairs); torch.cuda.synchronize(); ts.append(time.time() - t)
        res[r] = {"pairs": len(pairs), "seconds_median": round(statistics.median(ts), 2),
                  "pairs_per_second": round(len(pairs) / statistics.median(ts), 1), "runs_s": [round(x, 2) for x in ts]}
        log(f"[timing] {r}: {res[r]}")
        del fn
        torch.cuda.empty_cache()
    json.dump({"gpu": torch.cuda.get_device_name(0), "note": "model loaded before timing; warm-up excluded",
               "results": res}, open(out(name, "timing.json"), "w"), indent=1)


def task_verdict(name, corpus, claims, part="all"):
    """part: "llm" (PubGround judges), "ext" (VeriSci / MultiVerS, run in a fresh process) or "all"."""
    from run_baselines_verdict import VeriSci, MultiVerS
    oracle = [(c, d) for c, x in claims.items() for d in x["cand"]] if name != "scifact_open" else \
        [(c, d) for c, x in claims.items() for d in sorted(x["gold_ids"])]
    opn = []
    if name != "healthver":
        run = load_run(name, "PubGround hybrid + MedCPT")
        opn = [(c, d) for c in claims for d, _ in run[c][:3]]
    pairs = list(dict.fromkeys(oracle + opn))
    p = out(name, "verdict_preds.json")
    store = json.load(open(p)) if os.path.exists(p) else {}

    def sents(d):
        a = corpus[d]["abstract"]
        return list(a) if isinstance(a, list) else [a]
    for key in (["claude-haiku-5.5", "deepseek-v4.1-flash-1017", "claude-sonnet-5.5", "qwen3-8b"]
                if part in ("llm", "all") else []):
        call, llm = cached_call(key)
        judge = VerdictJudge(call, concurrency=16)
        res = asyncio.run(judge.judge_many([(claims[c]["claim"], corpus[d]["title"], sents(d)) for c, d in pairs]))
        store[f"PubGround judge: {key}"] = {f"{c}|{d}": [v["label"], v["rationale"]] for (c, d), v in zip(pairs, res)}
        log(f"[verdict {name}] {key}: new calls={llm.n_calls} cost=${llm.cost:.3f}")
        json.dump(store, open(p, "w"))  # judges are saved before the verifier models load
    ext = [("VeriSci", VeriSci, ()), ("MultiVerS-zs (fever_sci)", MultiVerS, ("fever_sci",)),
           ("MultiVerS-scifact (supervised, SciFact train+dev)", MultiVerS, ("scifact",))]
    if name == "healthver":
        ext.append(("MultiVerS-healthver (supervised, HealthVer train)", MultiVerS, ("healthver",)))
    for vname, cls, a in (ext if part in ("ext", "all") else []):
        have = store.setdefault(vname, {})
        missing = [(c, d) for c, d in pairs if f"{c}|{d}" not in have]
        if missing:
            m = cls(*a)
            for c, d in missing:
                have[f"{c}|{d}"] = list(m.predict(claims[c]["claim"], corpus[d]["title"], sents(d)))
            if hasattr(m, "close"):
                m.close()
            del m
        json.dump(store, open(p, "w"))
    json.dump({"oracle": oracle, "open": opn}, open(out(name, "verdict_pairs.json"), "w"))
    json.dump(store, open(p, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scifact", "scifact_open", "healthver"])
    ap.add_argument("--task", required=True, choices=["index", "pubground", "rerankers", "timing", "verdict"])
    ap.add_argument("--part", default="all", choices=["all", "llm", "ext"], help="verdict: judges, verifiers or both")
    a = ap.parse_args()
    corpus, claims = load(a.dataset)
    log(f"{a.dataset}: corpus={len(corpus)} claims={len(claims)}")
    if a.task == "index":
        corpus_obj(a.dataset, corpus)
    elif a.task == "pubground":
        task_pubground(a.dataset, corpus, claims)
    elif a.task == "rerankers":
        task_rerankers(a.dataset, corpus, claims)
    elif a.task == "timing":
        task_timing(a.dataset, corpus, claims)
    else:
        task_verdict(a.dataset, corpus, claims, a.part)


if __name__ == "__main__":
    main()
