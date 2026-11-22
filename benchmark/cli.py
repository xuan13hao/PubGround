"""
pubground-bench: one entry point for the PubGround benchmark.

  pubground-bench download                      # datasets + verifier checkpoints into $PUBGROUND_BENCH_DATA
  pubground-bench run [--dataset D]             # every PubGround / baseline system (GPU; LLM key for judges)
  pubground-bench eval-reranker --model HF_ID   # score YOUR cross-encoder on PubGround's candidates and compare
  pubground-bench eval-verdict --predictions F  # score YOUR verdict predictions and compare
  pubground-bench export --out DIR              # all systems' predictions in a release layout (Zenodo / HF)

Data directory: $PUBGROUND_BENCH_DATA (default: <package>/data).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# Source checkout: benchmark/data; installed package: ~/.cache/pubground-bench (override: PUBGROUND_BENCH_DATA)
DATA = os.environ.get("PUBGROUND_BENCH_DATA") or (
    os.path.join(HERE, "data") if "site-packages" not in HERE
    else os.path.join(os.path.expanduser("~"), ".cache", "pubground-bench"))
DATASETS = ["scifact", "scifact_open", "healthver"]
S3 = "https://scifact.s3-us-west-2.amazonaws.com"
S3B = "https://scifact.s3.us-west-2.amazonaws.com"
DOWNLOADS = [  # (url, destination dir under DATA) -- tarballs are extracted
    (f"{S3}/release/latest/data.tar.gz", "scifact_raw"),
    (f"{S3B}/scifact-open/latest/scifact_open.tar.gz", "scifact_open"),
    (f"{S3B}/longchecker/latest/data.tar.gz", "ext_multivers/data"),
    (f"{S3}/release/latest/models/rationale_roberta_large_scifact.tar.gz", "ext_verisci"),
    (f"{S3}/release/latest/models/label_roberta_large_fever_scifact.tar.gz", "ext_verisci"),
] + [(f"{S3B}/longchecker/latest/checkpoints/{n}.ckpt", "ext_multivers/ckpt") for n in ("fever_sci", "scifact", "healthver")]
BEIR = [("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", "corpus.parquet"),
        ("BeIR/scifact", "queries/queries-00000-of-00001.parquet", "queries.parquet"),
        ("BeIR/scifact-qrels", "test.tsv", "qrels_test.tsv")]


def _env():
    return dict(os.environ, PUBGROUND_BENCH_DATA=DATA)


def _script(name, *args):
    cmd = [sys.executable, os.path.join(HERE, name), *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=_env())


# ----------------------------- download -----------------------------
def cmd_download(a):
    os.makedirs(DATA, exist_ok=True)
    from huggingface_hub import hf_hub_download
    for repo, fn, local in BEIR:
        dst = os.path.join(DATA, local)
        if not os.path.exists(dst):
            shutil.copy(hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset"), dst)
    for url, sub in DOWNLOADS:
        dst_dir = os.path.join(DATA, sub)
        os.makedirs(dst_dir, exist_ok=True)
        name = url.rsplit("/", 1)[1]
        marker = os.path.join(dst_dir, f".done_{name}")
        if os.path.exists(marker):
            continue
        print(f"downloading {url}", flush=True)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with urllib.request.urlopen(url) as r:
                shutil.copyfileobj(r, tmp)
        if name.endswith(".tar.gz"):
            with tarfile.open(tmp.name) as tf:
                tf.extractall(dst_dir, filter="data")
            os.unlink(tmp.name)
        else:
            shutil.move(tmp.name, os.path.join(dst_dir, name))
        open(marker, "w").close()
    print(f"data ready in {DATA}")


# ----------------------------- run -----------------------------
def cmd_run(a):
    for ds in a.dataset or DATASETS:
        tasks = ["verdict"] if ds == "healthver" else ["index", "pubground", "rerankers", "timing", "verdict"]
        for t in tasks:
            if ds == "scifact_open" and t == "timing":
                continue  # throughput is measured once, on SciFact
            if t == "verdict":  # LLM judges and the verifier models run in separate processes
                _script("run_pubground.py", "--dataset", ds, "--task", t, "--part", "llm")
                _script("run_pubground.py", "--dataset", ds, "--task", t, "--part", "ext")
            else:
                _script("run_pubground.py", "--dataset", ds, "--task", t)
    _script("pubground_report.py")


# ----------------------------- metrics -----------------------------
def _ndcg10(ranked, gold):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))


def _gold(ds):
    sys.path.insert(0, HERE)
    os.environ["PUBGROUND_BENCH_DATA"] = DATA
    from run_pubground import load
    corpus, claims = load(ds)
    return corpus, claims


def _load_run(ds, name):
    p = os.path.join(DATA, "pubground", ds, "runs", f"{name}.json")
    if not os.path.exists(p):
        sys.exit(f"missing {p}: run `pubground-bench run --dataset {ds}` or download the released predictions")
    return {q: [str(d) for d, _ in v] for q, v in json.load(open(p)).items()}


def _paired(a, b, B=2000, seed=0):
    import numpy as np
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    reps = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def cmd_eval_reranker(a):
    """Score a Hugging Face cross-encoder on PubGround's hybrid top-100 and compare with every saved reranker."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    corpus, claims = _gold(a.dataset)
    cand = _load_run(a.dataset, "PubGround hybrid")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
    sys.path.insert(0, os.path.dirname(HERE))
    from pubmed_sentence_matcher.retrieval import doc_text
    run = {}
    for q, docs in cand.items():
        scores = []
        for i in range(0, len(docs), a.batch_size):
            enc = tok([claims[q]["claim"]] * len(docs[i:i + a.batch_size]),
                      [doc_text(corpus[d]) for d in docs[i:i + a.batch_size]],
                      truncation="only_second", padding=True, max_length=512, return_tensors="pt").to(a.device)
            with torch.no_grad():
                lo = model(**enc).logits
            scores += (lo[:, 0] if lo.shape[-1] == 1 else lo[:, -1] - lo[:, 0]).float().cpu().tolist()
        run[q] = [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])]
    qs = [q for q in claims if claims[q]["gold_ids"]]
    mine = [_ndcg10(run[q], claims[q]["gold_ids"]) for q in qs]
    print(f"{a.model} on {a.dataset}: nDCG@10 = {sum(mine) / len(mine):.4f}  ({len(qs)} claims)")
    runs_dir = os.path.join(DATA, "pubground", a.dataset, "runs")
    for f in sorted(os.listdir(runs_dir)):
        if f.startswith("PubGround hybrid ->") or f == "PubGround hybrid + MedCPT.json":
            other = _load_run(a.dataset, f[:-5])
            o = [_ndcg10(other[q], claims[q]["gold_ids"]) for q in qs]
            d, lo, hi = _paired(mine, o)
            print(f"  vs {f[:-5]:45s} {sum(o) / len(o):.4f}   paired diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def _label_f1(items):
    g = sum(x[0] for x in items); p = sum(x[1] for x in items); c = sum(x[2] for x in items)
    P = c / p if p else 0.0
    R = c / g if g else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


def cmd_eval_verdict(a):
    """predictions: JSONL rows {"claim_id", "doc_id", "label"} for the chosen input set."""
    import numpy as np
    corpus, claims = _gold(a.dataset)
    pairs = json.load(open(os.path.join(DATA, "pubground", a.dataset, "verdict_pairs.json")))[a.input]
    pairs = [(str(c), str(d)) for c, d in pairs]
    mine = {(str(r["claim_id"]), str(r["doc_id"])): str(r["label"]).upper() for r in map(json.loads, open(a.predictions))}
    missing = [p for p in pairs if p not in mine]
    if missing:
        sys.exit(f"{len(missing)} required (claim_id, doc_id) pairs are missing, e.g. {missing[:3]}")
    store = json.load(open(os.path.join(DATA, "pubground", a.dataset, "verdict_preds.json")))

    def items(pred):
        by = {}
        for c, d in pairs:
            by.setdefault(c, []).append(d)
        out = []
        for c, x in claims.items():
            gold = {str(d): (ev[0]["label"] if isinstance(ev, list) else ev["label"]) for d, ev in x["evidence"].items()}
            n_p = n_c = 0
            for d in by.get(c, []):
                lab = pred(c, d)
                if lab not in ("SUPPORT", "CONTRADICT"):
                    continue
                n_p += 1
                n_c += gold.get(d) == lab
            out.append((len(gold), n_p, n_c))
        return out
    m_items = items(lambda c, d: mine[(c, d)])
    print(f"your predictions on {a.dataset} ({a.input}): label F1 = {_label_f1(m_items):.4f}")
    rng_items = list(range(len(m_items)))
    for name, preds in store.items():
        o_items = items(lambda c, d: preds[f"{c}|{d}"][0])
        rng = np.random.default_rng(0)
        reps = []
        for _ in range(2000):
            idx = rng.integers(0, len(rng_items), len(rng_items))
            reps.append(_label_f1([m_items[i] for i in idx]) - _label_f1([o_items[i] for i in idx]))
        lo, hi = np.percentile(reps, [2.5, 97.5])
        print(f"  vs {name:52s} {_label_f1(o_items):.4f}   paired diff {_label_f1(m_items) - _label_f1(o_items):+.4f} [{lo:+.4f}, {hi:+.4f}]")


def cmd_export(a):
    _script("export_release.py", "--out", a.out)


def main():
    ap = argparse.ArgumentParser(prog="pubground-bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    r = sub.add_parser("run"); r.add_argument("--dataset", action="append", choices=DATASETS); r.set_defaults(fn=cmd_run)
    e = sub.add_parser("eval-reranker"); e.add_argument("--model", required=True)
    e.add_argument("--dataset", default="scifact", choices=["scifact", "scifact_open"])
    e.add_argument("--device", default="cuda"); e.add_argument("--batch-size", type=int, default=32)
    e.set_defaults(fn=cmd_eval_reranker)
    v = sub.add_parser("eval-verdict"); v.add_argument("--predictions", required=True)
    v.add_argument("--dataset", default="scifact", choices=DATASETS)
    v.add_argument("--input", default="oracle", choices=["oracle", "open"], help="gold abstracts or PubGround top-3")
    v.set_defaults(fn=cmd_eval_verdict)
    x = sub.add_parser("export"); x.add_argument("--out", required=True); x.set_defaults(fn=cmd_export)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
