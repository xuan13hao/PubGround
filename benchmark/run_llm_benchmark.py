"""
Full-pipeline benchmark on SciFact: the LLM stages, via OpenRouter.

`run_benchmark.py` measures the key-free part (BM25 / MedCPT dense / MedCPT rerank).
This script adds everything that needs an LLM or that the roadmap promises, on the same
300 SciFact test claims and 5,183-abstract corpus:

  Retrieval (BEIR metrics: nDCG@10, Recall@10/100, MRR@10, P@1)
    R1  Hybrid RRF(BM25, MedCPT dense) -> MedCPT rerank          (roadmap, no LLM)
    R2  Stage 1  LLM query optimization (pipeline prompt, verbatim) -> BM25 per query, RRF
    R3  Stage 1 + Stage 4.5 rerank                                (= the pipeline's retrieval path)
    R4  Stage 1 + hybrid + rerank

  Verdict (SciFact official abstract-level metrics; roadmap SUPPORTS/REFUTES/NEI stage)
    V1  Oracle: claim + its cited abstract -> SUPPORT / CONTRADICT / NEI (3-class acc, macro-F1)
    V2  Open:   claim + top-3 abstracts of the full pipeline (R3) -> label + rationale
        sentences; abstract-level Label-only and Rationale-selected P/R/F1.

All LLM responses are cached (data/llm_cache), MedCPT scores too (data/rerank_cache.pt),
so the run is resumable and reruns are free.

Usage:
  python benchmark/run_llm_benchmark.py --models anthropic/claude-haiku-5.5,deepseek/deepseek-v4.1-flash-1017
  python benchmark/run_llm_benchmark.py --max-queries 20 ...      # smoke test
"""
import argparse, json, math, os, re, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_benchmark import (load_data, metrics_for, aggregate, run_bm25, tok,  # noqa: E402
                           run_medcpt_dense, _device, log)
from llm_client import make_llm as LLM, parse_json  # noqa: E402

DATA = os.path.join(HERE, "data")
RRF_K = 60
# Models whose OpenRouter endpoint refuses reasoning={"enabled": false}.
REASONING_MANDATORY = ("anthropic/claude-sonnet-5.5",)


# ----------------------------- helpers -----------------------------
def rrf(run_lists, topk):
    """Reciprocal Rank Fusion of several ranked lists [(doc, score), ...]."""
    s = defaultdict(float)
    for lst in run_lists:
        for r, (d, _) in enumerate(lst):
            s[d] += 1.0 / (RRF_K + r + 1)
    return sorted(s.items(), key=lambda x: -x[1])[:topk]


def evaluate(runs, qrels):
    return aggregate([metrics_for([d for d, _ in runs[q]], qrels[q]) for q in runs])


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------- MedCPT rerank with cache -----------------------------
class Reranker:
    def __init__(self, docs, test_q, bs=64):
        self.docs, self.test_q, self.bs = docs, test_q, bs
        self.path = f"{DATA}/rerank_cache.json"
        self.cache = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self.mod = None

    def _load(self):
        if self.mod is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.dev = _device()
            log(f"[rerank] loading MedCPT cross-encoder on {self.dev} ...")
            self.tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
            self.mod = AutoModelForSequenceClassification.from_pretrained(
                "ncbi/MedCPT-Cross-Encoder").to(self.dev).eval()
            if self.dev == "cuda":
                self.mod = self.mod.half()

    def score_missing(self, cand):
        """cand: {qid: [doc ids]} -> fills cache for every (qid, doc) not yet scored."""
        import torch
        todo = [(q, d) for q, ds in cand.items() for d in ds if f"{q}|{d}" not in self.cache]
        if not todo:
            return
        self._load()
        log(f"[rerank] scoring {len(todo)} new pairs ...")
        for i in range(0, len(todo), self.bs):
            batch = todo[i:i + self.bs]
            pairs = [[self.test_q[q], (self.docs[d][0] + " " + self.docs[d][1])[:2000]]
                     for q, d in batch]
            enc = self.tok(pairs, truncation=True, padding=True, max_length=512,
                           return_tensors="pt").to(self.dev)
            with torch.no_grad():
                lo = self.mod(**enc).logits.squeeze(-1).float().cpu().tolist()
            for (q, d), s in zip(batch, lo if isinstance(lo, list) else [lo]):
                self.cache[f"{q}|{d}"] = s
            if (i // self.bs) % 200 == 0:
                log(f"    {i + len(batch)}/{len(todo)}")
        json.dump(self.cache, open(self.path, "w"))

    def rerank(self, runs, topk):
        cand = {q: [d for d, _ in runs[q][:topk]] for q in runs}
        self.score_missing(cand)
        return {q: sorted(((d, self.cache[f"{q}|{d}"]) for d in ds), key=lambda x: -x[1])
                for q, ds in cand.items()}


# ----------------------------- Stage 1: LLM query optimization -----------------------------
def stage1_messages(sentence):
    # Verbatim from PubMedSentenceMatcher._generate_optimized_queries
    return [
        {"role": "system",
         "content": "You are a biomedical research expert specializing in PubMed search query optimization."},
        {"role": "user",
         "content": f"""Given this sentence: "{sentence}"

Generate 3-5 optimized PubMed search queries that would best find articles containing similar statements. Consider:
1. Key biomedical concepts and terminology
2. Synonyms and related terms
3. MeSH terms if applicable
4. Different phrasings of the same concept
5. Both broad and specific queries

Respond with ONLY a JSON array of query strings, no other text:
["query1", "query2", "query3"]"""},
    ]


_FIELD_TAG = re.compile(r"\[[^\]]*\]")
_BOOL = {"and", "or", "not"}


def pubmed_query_to_bow(q):
    """PubMed syntax -> bag of words for BM25 (drop [MeSH]/[tiab] tags and boolean ops)."""
    return [w for w in tok(_FIELD_TAG.sub(" ", q)) if w not in _BOOL]


def run_stage1(llm, test_q, bm25_index, topk):
    qids = list(test_q)
    # The pipeline uses max_tokens=500; verbose models (Boolean/MeSH queries) overflow it and
    # the truncated JSON fails to parse, so the benchmark allows 1200 and salvages the rest.
    resp = llm.map([stage1_messages(test_q[q]) for q in qids], temperature=0.3,
                   max_tokens=1200, tag="stage1", log=log)
    expanded, n_fallback = {}, 0
    for q, r in zip(qids, resp):
        qs = parse_json(r)
        if qs is None and r:  # truncated array -> keep every complete "..." element
            qs = [json.loads(f'"{m}"') for m in re.findall(r'"((?:[^"\\]|\\.)*)"\s*[,\]]', r)]
        if not (isinstance(qs, list) and qs and all(isinstance(x, str) for x in qs)):
            qs, n_fallback = [], n_fallback + 1
        expanded[q] = qs[:5]
    runs = {}
    for q in qids:
        lists = [bm25_index(tok(test_q[q]), topk)]  # the original claim always runs too
        lists += [bm25_index(pubmed_query_to_bow(x), topk) for x in expanded[q]]
        runs[q] = rrf([l for l in lists if l], topk)
    return runs, expanded, n_fallback


# ----------------------------- Verdict -----------------------------
def verdict_messages(claim, title, sentences):
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return [
        {"role": "system",
         "content": "You are an expert biomedical fact-checker. You judge whether a scientific abstract "
                    "supports or contradicts a claim, citing the exact evidence sentences."},
        {"role": "user",
         "content": f"""Claim: "{claim}"

Abstract title: "{title}"
Abstract sentences:
{numbered}

Decide whether this abstract SUPPORTS the claim, CONTRADICTS it, or provides NOT ENOUGH INFO (NEI).
Only choose SUPPORT or CONTRADICT if specific sentences in the abstract directly establish it.
If SUPPORT or CONTRADICT, list the indices of the minimal set of rationale sentences (at most 3).

Respond with ONLY valid JSON, no other text:
{{"label": "SUPPORT" | "CONTRADICT" | "NEI", "rationale": [sentence indices]}}"""},
    ]


def norm_label(x):
    x = str(x or "").upper()
    if x.startswith("SUPPORT"):
        return "SUPPORT"
    if x.startswith("CONTRADICT") or x.startswith("REFUTE"):
        return "CONTRADICT"
    return "NEI"


def run_verdict(llm, claims, sci_corpus, pairs):
    """pairs: list of (claim_id, doc_id) -> {(cid, did): (label, [sent idx])}"""
    jobs = [verdict_messages(claims[c]["claim"], sci_corpus[d]["title"], sci_corpus[d]["abstract"])
            for c, d in pairs]
    resp = llm.map(jobs, temperature=0.0, max_tokens=400, tag="verdict", log=log)
    out, n_fail = {}, 0
    for (c, d), r in zip(pairs, resp):
        v = parse_json(r)
        if not isinstance(v, dict):
            n_fail += 1
            v = {}
        rat = v.get("rationale") or []
        rat = [int(i) for i in rat if str(i).lstrip("-").isdigit()] if isinstance(rat, list) else []
        out[(c, d)] = (norm_label(v.get("label")), rat)
    return out, n_fail


def scifact_abstract_metrics(claims, preds):
    """Official SciFact abstract-level evaluation (Wadden et al. 2020).
    preds: {cid: {did: (label, rationale)}}; NEI predictions are ignored; at most
    3 rationale sentences count. Gold = docs with evidence for that claim."""
    tp_label = tp_rat = n_pred = n_gold = 0
    for c, cl in claims.items():
        gold = {int(d): ev for d, ev in cl["evidence"].items()}
        n_gold += len(gold)
        for d, (lab, rat) in preds.get(c, {}).items():
            if lab == "NEI":
                continue
            n_pred += 1
            if d in gold and gold[d][0]["label"] == lab:
                tp_label += 1
                if any(set(g["sentences"]) <= set(rat[:3]) for g in gold[d]):
                    tp_rat += 1

    def prf(tp):
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gold if n_gold else 0.0
        return {"P": round(p, 4), "R": round(r, 4), "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0}

    return {"label_only": prf(tp_label), "rationale_selected": prf(tp_rat),
            "n_pred_evidence": n_pred, "n_gold_evidence": n_gold}


def three_class(gold, pred):
    labs = ["SUPPORT", "CONTRADICT", "NEI"]
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    f1s = {}
    for l in labs:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        pr, rc = (tp / (tp + fp) if tp + fp else 0), (tp / (tp + fn) if tp + fn else 0)
        f1s[l] = round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0
    conf = Counter(f"{g}->{p}" for g, p in zip(gold, pred))
    return {"n": len(gold), "accuracy": round(acc, 4),
            "macro_f1": round(sum(f1s.values()) / 3, 4), "per_class_f1": f1s,
            "confusion": dict(sorted(conf.items()))}


# ----------------------------- driver -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="anthropic/claude-haiku-5.5,deepseek/deepseek-v4.1-flash-1017")
    ap.add_argument("--verdict-models", default="",
                    help="extra models that run ONLY the verdict task (cheap: ~1.2k calls)")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--verdict-k", type=int, default=3, help="abstracts sent to the verdict stage")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=f"{HERE}/results_llm.json")
    args = ap.parse_args()
    t0 = time.time()

    docs, test_q, qrels = load_data()
    if args.max_queries:
        test_q = dict(list(test_q.items())[: args.max_queries])
        qrels = {q: qrels[q] for q in test_q}
    log(f"corpus={len(docs)} test_queries={len(test_q)} topk={args.topk}")

    sci_corpus = {}
    for line in open(f"{DATA}/scifact_raw/data/corpus.jsonl"):
        r = json.loads(line)
        sci_corpus[r["doc_id"]] = r
    claims = {}
    for line in open(f"{DATA}/scifact_raw/data/claims_dev.jsonl"):
        r = json.loads(line)
        if str(r["id"]) in test_q:
            claims[str(r["id"])] = r
    assert len(claims) == len(test_q), "SciFact dev claims must match BEIR test queries"

    results, info, timing = {}, {}, {}

    # ---- key-free systems (recomputed so every row comes from one run) ----
    from rank_bm25 import BM25Okapi
    import numpy as np
    doc_ids = list(docs)
    bm25 = BM25Okapi([tok(docs[d][0] + " " + docs[d][1]) for d in doc_ids])

    def bm25_index(toks, k):
        if not toks:
            return []
        sc = bm25.get_scores(toks)
        idx = np.argsort(sc)[::-1][:k]
        return [(doc_ids[i], float(sc[i])) for i in idx]

    bm25_runs = {q: bm25_index(tok(test_q[q]), args.topk) for q in test_q}
    results["BM25"] = evaluate(bm25_runs, qrels)

    dense_path = f"{DATA}/dense_runs_top{args.topk}.json"
    if os.path.exists(dense_path) and not args.max_queries:
        dense_runs = json.load(open(dense_path))
    else:
        dense_runs = run_medcpt_dense(docs, test_q, args.topk)
        if not args.max_queries:
            json.dump(dense_runs, open(dense_path, "w"))
    dense_runs = {q: dense_runs[q] for q in test_q}
    results["MedCPT dense"] = evaluate(dense_runs, qrels)

    rr = Reranker(docs, test_q)
    t = time.time()
    results["BM25 + MedCPT rerank"] = evaluate(rr.rerank(bm25_runs, args.topk), qrels)
    hybrid = {q: rrf([bm25_runs[q], dense_runs[q]], args.topk) for q in test_q}
    results["Hybrid RRF (BM25+dense)"] = evaluate(hybrid, qrels)
    hybrid_rr = rr.rerank(hybrid, args.topk)
    results["Hybrid RRF + MedCPT rerank"] = evaluate(hybrid_rr, qrels)
    timing["rerank (key-free)"] = round(time.time() - t, 1)
    log(json.dumps(results, indent=1))

    # ---- LLM stages, per model ----
    verdict_inputs = {}  # model -> {cid: [top doc ids]}
    for model in [m for m in args.models.split(",") if m]:
        llm = LLM(model, workers=args.workers,
                  reasoning=None if model in REASONING_MANDATORY else False)
        tag = model.split("/")[-1]
        t = time.time()
        qe_runs, expanded, n_fb = run_stage1(llm, test_q, bm25_index, args.topk)
        results[f"BM25 + Stage1 LLM-QE [{tag}]"] = evaluate(qe_runs, qrels)
        qe_rr = rr.rerank(qe_runs, args.topk)
        results[f"BM25 + Stage1 LLM-QE + rerank [{tag}]"] = evaluate(qe_rr, qrels)
        qe_hyb_rr = rr.rerank({q: rrf([qe_runs[q], dense_runs[q]], args.topk) for q in test_q}, args.topk)
        results[f"Hybrid(LLM-QE, dense) + rerank [{tag}]"] = evaluate(qe_hyb_rr, qrels)
        json.dump(expanded, open(f"{DATA}/stage1_queries_{tag}.json", "w"), indent=1)
        info[tag] = {"stage1_parse_fallbacks": n_fb,
                     "stage1_mean_queries": round(sum(len(v) for v in expanded.values()) / len(expanded), 2)}
        timing[f"LLM stages [{tag}]"] = round(time.time() - t, 1)
        # The full pipeline is Stage 1 -> BM25 -> MedCPT rerank; its top-k goes to the verdict.
        verdict_inputs[model] = {q: [d for d, _ in qe_rr[q][:args.verdict_k]] for q in test_q}
        log(f"[{tag}] cost so far ${llm.cost:.3f}")
        json.dump(rr.cache, open(rr.path, "w"))

    # Saved so run_baselines_verdict.py can feed external verifiers the identical abstracts.
    if not args.max_queries:
        vi_path = f"{DATA}/verdict_inputs.json"
        vi = json.load(open(vi_path)) if os.path.exists(vi_path) else {}
        vi.update({m.split("/")[-1]: v for m, v in verdict_inputs.items()})  # merge, keep other models
        json.dump(vi, open(vi_path, "w"))

    # ---- Verdict ----
    verdict = {}
    oracle_pairs = [(c, d) for c, cl in claims.items() for d in cl["cited_doc_ids"]]
    gold_oracle = []
    for c, d in oracle_pairs:
        ev = claims[c]["evidence"].get(str(d))
        gold_oracle.append(ev[0]["label"] if ev else "NEI")
    vmodels = [m for m in args.models.split(",") if m] + [m for m in args.verdict_models.split(",") if m]
    default_open = list(verdict_inputs.values())[0] if verdict_inputs else None
    for model in vmodels:
        llm = LLM(model, workers=args.workers,
                  reasoning=None if model in REASONING_MANDATORY else False)
        tag = model.split("/")[-1]
        # Open setting: this model's own full pipeline if it ran one, else the first model's.
        open_in = verdict_inputs.get(model, default_open)
        open_pairs = [(c, int(d)) for c in claims for d in open_in[c]]
        pv, n_fail = run_verdict(llm, claims, sci_corpus, sorted(set(oracle_pairs) | set(open_pairs)))
        pred_oracle = [pv[p][0] for p in oracle_pairs]
        open_preds = defaultdict(dict)
        for c, d in open_pairs:
            open_preds[c][d] = pv[(c, d)]
        orc_preds = defaultdict(dict)
        for c, d in oracle_pairs:
            orc_preds[c][d] = pv[(c, d)]
        verdict[tag] = {
            "oracle_3class": three_class(gold_oracle, pred_oracle),
            "oracle_abstract_level": scifact_abstract_metrics(claims, orc_preds),
            "open_abstract_level": scifact_abstract_metrics(claims, open_preds),
            "open_retrieval_from": (model if model in verdict_inputs
                                    else list(verdict_inputs)[0]).split("/")[-1],
            "parse_failures": n_fail,
        }
        log(f"[verdict {tag}] {json.dumps(verdict[tag]['oracle_3class'])}")

    # ---- cost ----
    cost = defaultdict(float)
    if os.path.exists(f"{DATA}/llm_usage.jsonl"):
        for line in open(f"{DATA}/llm_usage.jsonl"):
            r = json.loads(line)
            if r.get("tag") != "stage5":  # spend of the removed Stage 5 experiment is not counted
                cost[r["model"]] += float(r.get("cost") or 0)

    out = {
        "dataset": "SciFact (BEIR test = SciFact dev claims)",
        "corpus_size": len(docs), "n_queries": len(test_q), "topk": args.topk,
        "verdict_k": args.verdict_k,
        "models": args.models.split(","), "verdict_only_models": [m for m in args.verdict_models.split(",") if m],
        "retrieval": results, "verdict": verdict, "info": info, "timing_sec": timing,
        "llm_cost_usd_cumulative": {k: round(v, 4) for k, v in cost.items()},
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
