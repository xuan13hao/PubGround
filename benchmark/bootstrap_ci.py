"""
95% bootstrap confidence intervals (claims resampled with replacement, 2,000 replicates) for the
headline retrieval and verdict numbers, plus paired-difference CIs for the key comparisons.
Uses only cached runs / predictions (no new LLM calls). -> results_ci.json

  python benchmark/bootstrap_ci.py
"""
import json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_benchmark import load_data, metrics_for, log  # noqa: E402
from run_llm_benchmark import run_verdict, REASONING_MANDATORY  # noqa: E402
from llm_client import make_llm  # noqa: E402

DATA = f"{HERE}/data"
B = 2000
SEED = 0  # every statistic re-seeds, so adding a comparison never shifts the others
LLMS = ["anthropic/claude-haiku-5.5", "deepseek/deepseek-v4.1-flash-1017", "anthropic/claude-sonnet-5.5",
        "local:/N/project/mrust01/LLM_NER/models/qwen3-8b"]


def ci(vals_by_claim, stat):
    """vals_by_claim: list of per-claim items; stat(list_of_items) -> float."""
    n = len(vals_by_claim)
    point = stat(vals_by_claim)
    RNG = np.random.default_rng(SEED)
    reps = [stat([vals_by_claim[i] for i in RNG.integers(0, n, n)]) for _ in range(B)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def paired(a, b):
    """a, b: per-claim nDCG arrays (same claims). CI of mean(a - b) and bootstrap P(diff <= 0)."""
    d = np.array(a) - np.array(b)
    n = len(d)
    RNG = np.random.default_rng(SEED)
    reps = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p_le_0": round(float((reps <= 0).mean()), 4)}


# ----------------------------- retrieval -----------------------------
def ndcg_per_query(runs, qrels):
    return {q: metrics_for([str(d) for d, _ in runs[q]], qrels[q])["ndcg@10"] for q in qrels}


def retrieval():
    out = {}
    _, _, qrels = load_data()
    sf = {n: json.load(open(f"{DATA}/baseline_runs/{f}.json")) for n, f in [
        ("BM25", "BM25"), ("MedCPT dense", "MedCPT dense"), ("bge-large-en-v1.5", "bge-large-en-v1.5"),
        ("Qwen3-Embedding-4B", "Qwen3-Embedding-4B"),
        ("Hybrid + MedCPT", "Hybrid + MedCPT"), ("Hybrid + MiniLM-L12", "Hybrid + MiniLM-L12"),
        ("Hybrid + bge-reranker-v2-m3", "Hybrid + bge-reranker-v2-m3"), ("Hybrid + monoT5-3B", "Hybrid + monoT5-3B"),
        ("Hybrid + Qwen3-Reranker-4B", "Hybrid + Qwen3-Reranker-4B")] +
        [(f"LLM-QE + MedCPT [{t}]",) * 2 for t in ("claude-haiku-5.5", "deepseek-v4.1-flash-1017", "qwen3-8b")]}
    from run_llm_benchmark import rrf
    sf["Hybrid"] = {q: rrf([sf["BM25"][q], sf["MedCPT dense"][q]], 100) for q in qrels}
    so_q = {q: {str(d) for d in c["evidence"]} for q, c in
            ((str(r["id"]), r) for r in map(json.loads, open(f"{DATA}/scifact_open/data/claims.jsonl")))
            if c["evidence"]}
    so_dir = f"{DATA}/extra/scifact_open/runs"
    so = {os.path.basename(p)[:-5]: json.load(open(f"{so_dir}/{p}")) for p in os.listdir(so_dir)}
    for ds, runs, qr in (("scifact", sf, qrels), ("scifact_open", so, so_q)):
        per = {n: ndcg_per_query(r, qr) for n, r in runs.items()}
        qs = sorted(qr)
        out[ds] = {"ndcg@10": {n: ci([p[q] for q in qs], np.mean) for n, p in per.items()}}
        pairs = [("Hybrid + Qwen3-Reranker-4B", "Hybrid + MedCPT"), ("Hybrid + MedCPT", "Hybrid + monoT5-3B"),
                 ("Hybrid + MedCPT", "Hybrid + bge-reranker-v2-m3"), ("Hybrid + MedCPT", "Hybrid + MiniLM-L12"),
                 ("Hybrid + MedCPT", "Hybrid"), ("Hybrid + MedCPT", "BM25")]
        if ds == "scifact_open":
            pairs += [("LLM-QE + MedCPT [deepseek-v4.1-flash-1017]", "Hybrid + MedCPT"),
                      ("LLM-QE + MedCPT [claude-haiku-5.5]", "Hybrid + MedCPT"),
                      ("LLM-QE + MedCPT [qwen3-8b]", "Hybrid + MedCPT")]
        else:
            pairs += [("Qwen3-Embedding-4B", "MedCPT dense")]
        pairs += [(f"LLM-QE + MedCPT [{t}]", "Hybrid + MedCPT") for t in
                  ("deepseek-v4.1-flash-1017", "claude-haiku-5.5", "qwen3-8b") if ds == "scifact"]
        out[ds]["paired_ndcg@10"] = {f"{a} - {b}": paired([per[a][q] for q in qs], [per[b][q] for q in qs])
                                     for a, b in pairs}
        log(f"[retrieval {ds}] done")
    return out


# ----------------------------- verdict -----------------------------
def label_f1(items):
    """items: per-claim (n_gold, n_pred, n_correct) tuples -> abstract-level label-only F1."""
    g = sum(x[0] for x in items); p = sum(x[1] for x in items); c = sum(x[2] for x in items)
    P = c / p if p else 0.0
    R = c / g if g else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


def per_claim(claims, pairs, pv):
    by = defaultdict(list)
    for c, d in pairs:
        by[c].append(d)
    items = []
    for c, cl in claims.items():
        gold = {int(k): v[0]["label"] for k, v in cl["evidence"].items()}
        n_pred = n_cor = 0
        for d in by.get(c, []):
            lab = pv[(c, d)][0]
            if lab == "NEI":
                continue
            n_pred += 1
            n_cor += int(gold.get(int(d)) == lab)
        items.append((len(gold), n_pred, n_cor))
    return items


def paired_f1(items_a, items_b):
    """Paired bootstrap over claims of label-F1(a) - label-F1(b); items are per-claim tuples (same claims)."""
    n = len(items_a)
    d0 = label_f1(items_a) - label_f1(items_b)
    RNG = np.random.default_rng(SEED)
    reps = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        reps.append(label_f1([items_a[i] for i in idx]) - label_f1([items_b[i] for i in idx]))
    reps = np.array(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"diff": round(d0, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p_le_0": round(float((reps <= 0).mean()), 4)}


def diagnostics(pairs, gold, preds):
    """3-class confusion (gold -> predicted) and the NEI rate on gold SUPPORT/CONTRADICT pairs."""
    out = {}
    for k, pv in preds.items():
        conf = defaultdict(int)
        for p in pairs:
            conf[f"{gold.get(p, 'NEI')}->{pv[p][0]}"] += 1
        ev = [p for p in pairs if p in gold]
        out[k] = {"confusion": dict(sorted(conf.items())),
                  "nei_rate_on_gold_evidence": round(sum(pv[p][0] == "NEI" for p in ev) / len(ev), 4),
                  "n_gold_evidence_pairs": len(ev),
                  "nei_rate_overall": round(sum(pv[p][0] == "NEI" for p in pairs) / len(pairs), 4)}
    return out


API = ["LLM claude-haiku-5.5", "LLM deepseek-v4.1-flash-1017", "LLM claude-sonnet-5.5"]
JUDGES = API + ["LLM qwen3-8b"]


def verdict_preds(claims, corpus, pairs, store):
    preds = {k: {(c, d): tuple(v[f"{c}|{d}"]) for c, d in pairs} for k, v in store.items()}
    for m in LLMS:
        llm = make_llm(m, reasoning=None if m in REASONING_MANDATORY else False)
        pv, _ = run_verdict(llm, claims, corpus, pairs)
        assert llm.n_calls == 0, f"{m}: expected cached responses only"
        preds["LLM " + m.split("/")[-1]] = pv
    return preds


def verdict():
    out, paired, diag = {}, {}, {}
    # SciFact
    sci = {r["doc_id"]: r for r in map(json.loads, open(f"{DATA}/scifact_raw/data/corpus.jsonl"))}
    claims = {str(r["id"]): r for r in map(json.loads, open(f"{DATA}/scifact_raw/data/claims_dev.jsonl"))}
  # dict.fromkeys: SciFact dev claim 1245 lists cited doc 7662395 twice; count each pair once
    oracle = list(dict.fromkeys((c, d) for c, cl in claims.items() for d in cl["cited_doc_ids"]))
    vin = json.load(open(f"{DATA}/verdict_inputs.json"))["claude-haiku-5.5"]
    opn = [(c, int(d)) for c in claims for d in vin[c]]
    store = json.load(open(f"{DATA}/verdict_baseline_preds.json"))
    pr = verdict_preds(claims, sci, sorted(set(oracle) | set(opn)), store)
    out["scifact_oracle"] = {k: ci(per_claim(claims, oracle, v), label_f1) for k, v in pr.items()}
    out["scifact_open"] = {k: ci(per_claim(claims, opn, v), label_f1) for k, v in pr.items()}
    paired, diag = {}, {}
    it = {k: per_claim(claims, oracle, v) for k, v in pr.items()}
    paired["scifact_oracle"] = {f"{j} - {b}": paired_f1(it[j], it[b]) for j in JUDGES
                                for b in ("VeriSci", "MultiVerS-zs (fever_sci)")}
    it = {k: per_claim(claims, opn, v) for k, v in pr.items()}
    paired["scifact_open"] = {f"{j} - VeriSci": paired_f1(it[j], it["VeriSci"]) for j in JUDGES}
    gold = {(c, int(d)): ev[0]["label"] for c, cl in claims.items() for d, ev in cl["evidence"].items()}
    diag["scifact_oracle"] = diagnostics(oracle, gold, pr)
    log("[verdict scifact] done")
    # SciFact-Open and HealthVer
    from run_extra_datasets import load_dataset, load_run
    for ds in ("scifact_open", "healthver"):
        corpus, cl = load_dataset(ds)
        if ds == "healthver":
            pairs = [(c, d) for c, x in cl.items() for d in x["cand"]]
            key = "healthver"
        else:
            rr = load_run(ds, "Hybrid + MedCPT")
            pairs = [(c, d) for c in cl for d, _ in rr[c][:3]]
            key = "scifact_open_open"
        orc = [(c, int(d)) for c, x in cl.items() for d in x["evidence"]] if ds == "scifact_open" else []
        store = json.load(open(f"{DATA}/extra/{ds}/verdict_preds.json"))
        allp = sorted(set(pairs) | set(orc))
        pr = verdict_preds(cl, corpus, allp, store)
        out[key] = {k: ci(per_claim(cl, pairs, v), label_f1) for k, v in pr.items()}
        it = {k: per_claim(cl, pairs, v) for k, v in pr.items()}
        gold = {(c, int(d)): (ev[0]["label"] if isinstance(ev, list) else ev["label"])
                for c, x in cl.items() for d, ev in x["evidence"].items()}
        if ds == "healthver":
            sup = "MultiVerS-healthver (supervised, HealthVer train)"
            paired[key] = {f"{sup} - {j}": paired_f1(it[sup], it[j]) for j in JUDGES}
            diag[key] = diagnostics(pairs, gold, pr)
        else:
            sup = "MultiVerS-scifact (supervised, SciFact train+dev)"
            paired[key] = {f"{j} - {b}": paired_f1(it[j], it[b]) for j in JUDGES for b in (sup, "VeriSci")}
            diag["scifact_open_gold_evidence"] = diagnostics(orc, gold, pr)
        log(f"[verdict {ds}] done")
    return out, paired, diag


def main():
    v, vp, vd = verdict()
    res = {"method": f"nonparametric bootstrap over claims, {B} replicates, 95% percentile intervals, seed 0 per statistic",
           "retrieval": retrieval(), "verdict_label_f1": v, "verdict_paired_label_f1": vp,
           "verdict_diagnostics": vd}
    json.dump(res, open(f"{HERE}/results_ci.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
