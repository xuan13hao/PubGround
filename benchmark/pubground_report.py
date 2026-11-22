"""
Results of the PubGround benchmark (run_pubground.py outputs) -> results_pubground.json + PUBGROUND_RESULTS.md.

Retrieval: nDCG@10 / R@10 / R@100 / MRR@10 / P@1 with 95% bootstrap CIs and paired differences.
Verdict: abstract-level label F1 (gold and top-3 inputs) with CIs, paired differences, NEI rates and
HealthVer confusion matrices. Bootstrap: claims resampled, 2,000 replicates, seed 0 per statistic.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_pubground import load, load_run, out  # noqa: E402

DATA = os.environ.get("PUBGROUND_BENCH_DATA") or os.path.join(HERE, "data")
B, SEED = 2000, 0
M = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "p@1"]
PG = "PubGround hybrid + MedCPT"
JUDGES = ["PubGround judge: claude-haiku-5.5", "PubGround judge: deepseek-v4.1-flash-1017",
          "PubGround judge: claude-sonnet-5.5", "PubGround judge: qwen3-8b"]
API = JUDGES[:3]


def metrics(ranked, gold):
    r = {}
    r["ndcg@10"] = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold) / \
        sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    for k in (10, 100):
        r[f"recall@{k}"] = len(gold & set(ranked[:k])) / len(gold)
    r["mrr@10"] = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in gold), 0.0)
    r["p@1"] = float(bool(ranked) and ranked[0] in gold)
    return r


def boot_mean(x):
    x = np.array(x)
    rng = np.random.default_rng(SEED)
    reps = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": round(float(x.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def boot_diff(a, b):
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(SEED)
    reps = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p_le_0": round(float((reps <= 0).mean()), 4)}


def external_first_stage(ds):
    """Dense retrievers that do not depend on PubGround (from the earlier baseline runs)."""
    if ds == "scifact":
        base, names = f"{DATA}/baseline_runs", ["e5-large-v2", "bge-large-en-v1.5", "SPLADE", "Qwen3-Embedding-4B"]
    else:
        base, names = f"{DATA}/extra/scifact_open/runs", ["e5-large-v2", "bge-large-en-v1.5", "Qwen3-Embedding-4B"]
    runs = {}
    for n in names:
        p = f"{base}/{n}.json"
        if os.path.exists(p):
            runs[n] = {q: [str(d) for d, _ in v] for q, v in json.load(open(p)).items()}
    return runs


def retrieval(ds):
    corpus, claims = load(ds)
    qs = [q for q in claims if claims[q]["gold_ids"]]
    runs = {}
    for f in sorted(os.listdir(out(ds, "runs", ""))):
        runs[f[:-5]] = {q: [d for d, _ in v] for q, v in load_run(ds, f[:-5]).items()}
    runs.update({f"{k} (external)": v for k, v in external_first_stage(ds).items()})
    per = {n: {q: metrics(r[q], claims[q]["gold_ids"]) for q in qs} for n, r in runs.items()}
    table = {n: {m: round(float(np.mean([p[q][m] for q in qs])), 4) for m in M} for n, p in per.items()}
    ci = {n: boot_mean([p[q]["ndcg@10"] for q in qs]) for n, p in per.items()}
    nd = {n: [p[q]["ndcg@10"] for q in qs] for n, p in per.items()}
    paired = {}
    for n in nd:
        if n.startswith("PubGround hybrid ->") or n.startswith("OpenScholar") or n.startswith("PubGround Stage1"):
            paired[f"{PG} - {n}"] = boot_diff(nd[PG], nd[n])
    for n in ("PubGround hybrid", "PubGround BM25", "PubGround MedCPT dense"):
        paired[f"{PG} - {n}"] = boot_diff(nd[PG], nd[n])
    return {"n_claims": len(qs), "metrics": table, "ndcg_ci": ci, "paired_ndcg": paired}


def verdict(ds):
    corpus, claims = load(ds)
    store = json.load(open(out(ds, "verdict_preds.json")))
    sets = json.load(open(out(ds, "verdict_pairs.json")))
    gold = {}
    for c, x in claims.items():
        for d, ev in x["evidence"].items():
            gold[(c, str(d))] = ev[0]["label"] if isinstance(ev, list) else ev["label"]
    res = {}
    for inp in ("oracle", "open"):
        pairs = [(str(c), str(d)) for c, d in sets[inp]]
        if not pairs:
            continue
        by = defaultdict(list)
        for c, d in pairs:
            by[c].append(d)

        def items(pv):
            outl = []
            for c, x in claims.items():
                g = {(str(d)) for d in x["evidence"]}
                n_p = n_c = 0
                for d in by.get(c, []):
                    lab = pv[f"{c}|{d}"][0]
                    if lab not in ("SUPPORT", "CONTRADICT"):
                        continue
                    n_p += 1
                    n_c += gold.get((c, d)) == lab
                outl.append((len(g), n_p, n_c))
            return outl

        def f1(it):
            g = sum(x[0] for x in it); p = sum(x[1] for x in it); c = sum(x[2] for x in it)
            P = c / p if p else 0.0
            R = c / g if g else 0.0
            return 2 * P * R / (P + R) if P + R else 0.0, P, R

        def ci(it):
            rng = np.random.default_rng(SEED)
            reps = [f1([it[i] for i in rng.integers(0, len(it), len(it))])[0] for _ in range(B)]
            lo, hi = np.percentile(reps, [2.5, 97.5])
            F, P, R = f1(it)
            return {"F1": round(F, 4), "P": round(P, 4), "R": round(R, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}

        def pdiff(a, b):
            rng = np.random.default_rng(SEED)
            reps = []
            for _ in range(B):
                idx = rng.integers(0, len(a), len(a))
                reps.append(f1([a[i] for i in idx])[0] - f1([b[i] for i in idx])[0])
            reps = np.array(reps)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            return {"diff": round(f1(a)[0] - f1(b)[0], 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                    "p_le_0": round(float((reps <= 0).mean()), 4)}

        it = {k: items(v) for k, v in store.items()}
        r = {"n_pairs": len(pairs), "label_f1": {k: ci(v) for k, v in it.items()}, "paired": {}}
        for j in JUDGES:
            for base in [b for b in store if not b.startswith("PubGround")]:
                if ds == "scifact" and "scifact (supervised" in base:
                    continue  # trained on these claims
                r["paired"][f"{j} - {base}"] = pdiff(it[j], it[base])
        if inp == "oracle":
            labs = ("SUPPORT", "CONTRADICT", "NEI")
            ev = [p for p in pairs if p in gold]
            r["nei_rate_on_gold_evidence"] = {k: round(sum(store[k][f"{c}|{d}"][0] == "NEI" for c, d in ev) / len(ev), 4)
                                              for k in store}
            acc = {}
            for k in store:
                g = [gold.get(p, "NEI") for p in pairs]
                pr = [store[k][f"{c}|{d}"][0] for c, d in pairs]
                f1s = []
                for l in labs:
                    tp = sum(a == l and b == l for a, b in zip(g, pr)); fp = sum(a != l and b == l for a, b in zip(g, pr))
                    fn = sum(a == l and b != l for a, b in zip(g, pr))
                    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
                    f1s.append(2 * P * R / (P + R) if P + R else 0)
                conf = defaultdict(int)
                for a, b in zip(g, pr):
                    conf[f"{a}->{b}"] += 1
                acc[k] = {"accuracy": round(sum(a == b for a, b in zip(g, pr)) / len(g), 4),
                          "macro_f1": round(sum(f1s) / 3, 4), "confusion": dict(sorted(conf.items()))}
            r["three_class"] = acc
        res[inp] = r
    return res


def fmt(x):
    return f"{x:.3f}"


def main():
    R = {"method": f"bootstrap over claims, {B} replicates, 95% percentile intervals, seed {SEED} per statistic"}
    for ds in ("scifact", "scifact_open"):
        R[ds] = {"retrieval": retrieval(ds), "verdict": verdict(ds)}
        t = out(ds, "timing.json")
        if os.path.exists(t):
            R[ds]["timing"] = json.load(open(t))
    R["healthver"] = {"verdict": verdict("healthver")}
    json.dump(R, open(f"{HERE}/results_pubground.json", "w"), indent=1)

    L = ["# PubGround benchmark results\n", "Generated by `pubground_report.py` from `run_pubground.py` outputs. "
         "Every row starting with *PubGround* is produced by the PubGround package itself.\n"]
    for ds, title in (("scifact", "SciFact (300 claims, 5,183 abstracts)"),
                      ("scifact_open", "SciFact-Open (279 claims, 500,000 abstracts; 206 with evidence)")):
        r = R[ds]["retrieval"]
        L.append(f"## {title}\n\n### Retrieval\n")
        L.append("| System | nDCG@10 (95% CI) | R@10 | R@100 | MRR@10 | P@1 |\n|---|---|---|---|---|---|")
        for n, m in sorted(r["metrics"].items(), key=lambda x: -x[1]["ndcg@10"]):
            c = r["ndcg_ci"][n]
            L.append(f"| {n} | {fmt(m['ndcg@10'])} ({fmt(c['lo'])}–{fmt(c['hi'])}) | {fmt(m['recall@10'])} | "
                     f"{fmt(m['recall@100'])} | {fmt(m['mrr@10'])} | {fmt(m['p@1'])} |")
        L.append("\n**Paired nDCG@10 differences**\n\n| Comparison | Diff | 95% CI |\n|---|---|---|")
        for k, v in r["paired_ndcg"].items():
            L.append(f"| {k} | {v['diff']:+.3f} | {v['lo']:+.3f} to {v['hi']:+.3f} |")
        if "timing" in R[ds]:
            L.append(f"\n**Reranker throughput** ({R[ds]['timing']['gpu']}; model loaded, warm-up excluded)\n")
            L.append("| Reranker | Pairs/s | Seconds for 2,000 pairs (median of 3) |\n|---|---|---|")
            for k, v in R[ds]["timing"]["results"].items():
                L.append(f"| {k} | {v['pairs_per_second']} | {v['seconds_median']} |")
    for ds in ("scifact", "scifact_open", "healthver"):
        for inp, v in R[ds]["verdict"].items():
            L.append(f"\n### Verdict — {ds}, {'gold abstracts' if inp == 'oracle' else 'top-3 of ' + PG} ({v['n_pairs']} pairs)\n")
            L.append("| Verifier | Label F1 (95% CI) | P | R |" + (" 3-class acc | NEI rate on gold evidence |" if inp == "oracle" else "")
                     + "\n|---|---|---|---|" + ("---|---|" if inp == "oracle" else ""))
            for k, f in v["label_f1"].items():
                row = f"| {k} | {fmt(f['F1'])} ({fmt(f['lo'])}–{fmt(f['hi'])}) | {fmt(f['P'])} | {fmt(f['R'])} |"
                if inp == "oracle":
                    row += f" {fmt(v['three_class'][k]['accuracy'])} | {fmt(v['nei_rate_on_gold_evidence'][k])} |"
                L.append(row)
            L.append("\n| Paired label-F1 difference | Diff | 95% CI |\n|---|---|---|")
            for k, p in v["paired"].items():
                L.append(f"| {k} | {p['diff']:+.3f} | {p['lo']:+.3f} to {p['hi']:+.3f} |")
    open(f"{HERE}/PUBGROUND_RESULTS.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
