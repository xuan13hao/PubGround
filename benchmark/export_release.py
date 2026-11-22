"""
Export every system's predictions in a release layout for Zenodo / Hugging Face Datasets.

  python export_release.py --out ../pubground-bench-release

Layout
  retrieval/<dataset>/<system>.jsonl   {"claim_id", "ranking": [[doc_id, score], ...]}   (top-100, best first)
  verdict/<dataset>.jsonl              {"claim_id", "doc_id", "input", "system", "label", "rationale"}
  gold/<dataset>.jsonl                 {"claim_id", "claim", "doc_id", "label"}  (ids/labels from the public datasets)
  results/                             results_pubground.json, PUBGROUND_RESULTS.md
  README.md, CITATION.cff, .zenodo.json, LICENSE
Only identifiers, labels, scores and claim texts are exported, no abstract text.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_pubground import load, load_run, out  # noqa: E402

VERSION = "1.0.0"
TITLE = "PubGround-Bench: retrieval rankings and claim-verification predictions on SciFact, SciFact-Open and HealthVer"
REPO = "https://github.com/xuan13hao/citation_agents_system"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    o = ap.parse_args().out
    os.makedirs(o, exist_ok=True)
    counts = {}
    for ds in ("scifact", "scifact_open", "healthver"):
        corpus, claims = load(ds)
        os.makedirs(f"{o}/gold", exist_ok=True)
        with open(f"{o}/gold/{ds}.jsonl", "w") as f:
            for c, x in claims.items():
                docs = x["cand"] if ds == "healthver" else sorted(x["gold_ids"])
                if not docs:
                    f.write(json.dumps({"claim_id": c, "claim": x["claim"], "doc_id": None, "label": None}) + "\n")
                for d in docs:
                    ev = x["evidence"].get(d) or x["evidence"].get(int(d) if d.isdigit() else d)
                    lab = (ev[0]["label"] if isinstance(ev, list) else ev["label"]) if ev else "NEI"
                    f.write(json.dumps({"claim_id": c, "claim": x["claim"], "doc_id": d, "label": lab}) + "\n")
        runs_dir = out(ds, "runs", "")
        if os.path.isdir(runs_dir) and os.listdir(runs_dir):
            os.makedirs(f"{o}/retrieval/{ds}", exist_ok=True)
            for fn in sorted(os.listdir(runs_dir)):
                run = load_run(ds, fn[:-5])
                with open(f"{o}/retrieval/{ds}/{fn[:-5].replace(' ', '_').replace('>', '').replace('/', '-')}.jsonl", "w") as f:
                    for c, v in run.items():
                        f.write(json.dumps({"claim_id": c, "system": fn[:-5], "ranking": [[d, round(s, 6)] for d, s in v]}) + "\n")
                counts[f"retrieval/{ds}"] = counts.get(f"retrieval/{ds}", 0) + 1
        preds, sets = out(ds, "verdict_preds.json"), out(ds, "verdict_pairs.json")
        if os.path.exists(preds):
            store, pairs = json.load(open(preds)), json.load(open(sets))
            os.makedirs(f"{o}/verdict", exist_ok=True)
            with open(f"{o}/verdict/{ds}.jsonl", "w") as f:
                n = 0
                for inp, ps in pairs.items():
                    for c, d in ps:
                        for system, v in store.items():
                            lab, rat = v[f"{c}|{d}"]
                            f.write(json.dumps({"claim_id": str(c), "doc_id": str(d), "input": inp, "system": system,
                                                "label": lab, "rationale": rat}) + "\n")
                            n += 1
            counts[f"verdict/{ds}"] = n
    os.makedirs(f"{o}/results", exist_ok=True)
    for f in ("results_pubground.json", "PUBGROUND_RESULTS.md"):
        if os.path.exists(f"{HERE}/{f}"):
            shutil.copy(f"{HERE}/{f}", f"{o}/results/{f}")

    open(f"{o}/README.md", "w").write(f"""---
license: cc-by-4.0
pretty_name: PubGround-Bench
task_categories: [text-retrieval, text-classification]
language: [en]
tags: [biomedical, claim-verification, citation, retrieval, scifact, healthver]
---
# {TITLE}

Cached outputs of every system in the PubGround benchmark ({REPO}), so a new retriever, reranker or claim
verifier can be compared on identical inputs without re-running anything.

* `retrieval/<dataset>/<system>.jsonl` — top-100 abstract rankings per claim: PubGround configurations (BM25,
  MedCPT dense, hybrid RRF, hybrid + MedCPT, LLM query expansion), external rerankers scored on PubGround's
  hybrid candidates (MiniLM, bge-reranker-v2-m3, monoT5-3B, Qwen3-Reranker-4B, OpenScholar reranker) and the
  OpenScholar retriever pipeline.
* `verdict/<dataset>.jsonl` — SUPPORT / CONTRADICT / NEI predictions with rationale sentence indices for the
  PubGround judge (Claude Haiku 5.5, DeepSeek V4.1 Flash, Claude Sonnet 5.5, Qwen3-8B), VeriSci and MultiVerS,
  on gold abstracts (`input = oracle`) and on PubGround's top-3 abstracts (`input = open`).
* `gold/<dataset>.jsonl` — claim ids, claim texts, abstract ids and labels as used for scoring.
* `results/` — all metrics, 95% bootstrap intervals and paired comparisons.

Compare your own model with `pip install "pubground[bench]"` and `pubground-bench eval-reranker` /
`pubground-bench eval-verdict` (see the repository).

**Note.** The released MultiVerS SciFact checkpoint was trained on all 300 SciFact development claims (the BEIR
SciFact test claims); its predictions on SciFact are included for transparency but are not a valid comparison.

**Sources and licenses.** Gold labels and claim texts come from SciFact (Wadden et al., 2020), SciFact-Open
(Wadden et al., 2022) and HealthVer (Sarrouti et al., 2021; CC BY 4.0); please also follow those datasets'
licenses and cite them. Document ids refer to the original corpora; no abstract text is redistributed. The
predictions and results in this release are licensed CC BY 4.0.
""")
    open(f"{o}/LICENSE", "w").write("Predictions and results: Creative Commons Attribution 4.0 International (CC BY 4.0), "
                                    "https://creativecommons.org/licenses/by/4.0/\nGold labels and claim texts: see the "
                                    "upstream dataset licenses (SciFact, SciFact-Open, HealthVer).\n")
    open(f"{o}/CITATION.cff", "w").write(f"""cff-version: 1.2.0
message: "If you use these predictions, please cite PubGround and the underlying datasets."
title: "{TITLE}"
version: "{VERSION}"
authors:
  - family-names: Xuan
    given-names: Hao
license: CC-BY-4.0
repository-code: "{REPO}"
""")
    json.dump({"title": TITLE, "upload_type": "dataset", "version": VERSION, "license": "cc-by-4.0",
               "creators": [{"name": "Xuan, Hao"}],
               "description": "Cached retrieval rankings and claim-verification predictions of PubGround and "
                              "baseline systems on SciFact, SciFact-Open and HealthVer, with metrics and bootstrap "
                              "intervals. See README.md.",
               "keywords": ["biomedical retrieval", "claim verification", "SciFact", "HealthVer", "benchmark"],
               "related_identifiers": [{"identifier": REPO, "relation": "isSupplementTo", "scheme": "url"}]},
              open(f"{o}/.zenodo.json", "w"), indent=1)
    print(json.dumps(counts, indent=1), "->", o)


if __name__ == "__main__":
    main()
