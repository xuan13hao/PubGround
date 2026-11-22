"""
Layer 2 comparison: claim verification (SUPPORT / CONTRADICT / NEI + rationale sentences)
by this project's LLM verdict stage vs. published SciFact verifiers, on identical inputs.

Verifiers
  VeriSci        Wadden et al. 2020 (SciFact paper): RoBERTa-large rationale selector
                 (trained on SciFact train) + RoBERTa-large label predictor (FEVER + SciFact train).
  MultiVerS-zs   Wadden et al. 2022, `fever_sci` checkpoint: FEVER + PubMedQA + EvidenceInference,
                 never trained on SciFact -> zero-shot, like the LLMs.
  MultiVerS-sf*  `scifact` checkpoint. Its training file contains all 300 SciFact dev claims
                 (= the BEIR test claims used here), so it is CONTAMINATED; reported only to
                 show that, never as a comparable number.
  LLM            claude-haiku-5.5 / deepseek-v4.1-flash-1017 / claude-sonnet-5.5 with this
                 project's verdict prompt (responses come from the run_llm_benchmark cache).

Inputs (same for every verifier)
  oracle  each claim + its cited abstract(s)           (340 pairs)
  open    each claim + top-3 abstracts of the full pipeline (Haiku Stage1 -> BM25 -> MedCPT rerank),
          from data/verdict_inputs.json written by run_llm_benchmark.py

MultiVerS is run by re-implementing its inference (tokenization, global attention, heads,
thresholds) on current transformers; checkpoints are loaded with weights_only=True plus
inert stubs for the two pytorch-lightning metadata objects in the pickle.
"""
import argparse, json, os, sys, types
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_benchmark import log  # noqa: E402
from run_llm_benchmark import (scifact_abstract_metrics, three_class, run_verdict,  # noqa: E402
                               REASONING_MANDATORY)
from llm_client import make_llm as LLM  # noqa: E402

DATA = os.path.join(HERE, "data")
MV_CKPT = f"{DATA}/ext_multivers/ckpt"
VS_DIR = f"{DATA}/ext_verisci"


def dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- VeriSci -----------------------------
class VeriSci:
    LABELS = ["CONTRADICT", "NEI", "SUPPORT"]

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.rt = AutoTokenizer.from_pretrained(f"{VS_DIR}/rationale_roberta_large_scifact")
        self.rm = AutoModelForSequenceClassification.from_pretrained(
            f"{VS_DIR}/rationale_roberta_large_scifact").to(dev()).eval()
        self.lt = AutoTokenizer.from_pretrained(f"{VS_DIR}/label_roberta_large_fever_scifact")
        self.lm = AutoModelForSequenceClassification.from_pretrained(
            f"{VS_DIR}/label_roberta_large_fever_scifact", num_labels=3).to(dev()).eval()
        self.torch = torch

    def predict(self, claim, title, sentences):
        torch = self.torch
        with torch.no_grad():
            enc = self.rt(sentences, [claim] * len(sentences), padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(dev())
            p = torch.softmax(self.rm(**enc).logits, dim=1)[:, 1].cpu().numpy()
            rat = [int(i) for i in (p >= 0.5).nonzero()[0]]
            if not rat:
                return "NEI", []
            ev = " ".join(sentences[i] for i in rat)
            enc = self.lt(ev, claim, truncation="only_first", max_length=512, return_tensors="pt").to(dev())
            lab = self.LABELS[int(self.lm(**enc).logits.argmax(-1))]
        return lab, rat


# ----------------------------- MultiVerS -----------------------------
def _load_lightning_ckpt(path):
    """weights_only load; the two pytorch-lightning objects in the pickle become inert stubs."""
    import argparse as _ap
    import torch

    class ModelCheckpoint:  # noqa: D401 - stub, only receives pickled state
        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {})

    def _gpus_arg_default(x):
        return x

    for modname, attr, obj in [("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint", ModelCheckpoint),
                               ("pytorch_lightning.utilities.argparse", "_gpus_arg_default", _gpus_arg_default)]:
        parts = modname.split(".")
        for i in range(1, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
        setattr(sys.modules[modname], attr, obj)
        obj.__module__ = modname
        obj.__qualname__ = attr
    torch.serialization.add_safe_globals([_ap.Namespace, ModelCheckpoint, _gpus_arg_default])
    return torch.load(path, map_location="cpu", weights_only=True)


class MultiVerS:
    LABELS = {0: "CONTRADICT", 1: "NEI", 2: "SUPPORT"}
    ADD_TOKENS = ["<|sec|>", "</|sec|>", "<|sec-title|>", "</|sec-title|>", "<|abs|>", "</|abs|>",
                  "<|title|>", "</|title|>", "<|sent|>", "<|par|>"]

    def __init__(self, ckpt_name):
        import torch
        from torch import nn
        from transformers import AutoTokenizer, LongformerConfig, LongformerModel
        ck = _load_lightning_ckpt(f"{MV_CKPT}/{ckpt_name}.ckpt")
        hp = ck["hyper_parameters"]["hparams"]
        self.label_threshold = getattr(hp, "label_threshold", None)
        self.rationale_threshold = getattr(hp, "rationale_threshold", 0.5) or 0.5
        sd = ck["state_dict"]
        self.tok = AutoTokenizer.from_pretrained("allenai/longformer-large-4096")
        self.tok.add_tokens(self.ADD_TOKENS)
        cfg = LongformerConfig.from_pretrained("allenai/longformer-large-4096")
        cfg.vocab_size = sd["encoder.embeddings.word_embeddings.weight"].shape[0]
        enc = LongformerModel(cfg)
        missing, unexpected = enc.load_state_dict(
            {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        unexpected = [k for k in unexpected if "position_ids" not in k]
        assert not missing and not unexpected, (missing, unexpected)
        H = cfg.hidden_size

        def ff(prefix, din, dims):
            layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip([din] + dims[:-1], dims)])
            for i, l in enumerate(layers):
                l.weight.data = sd[f"{prefix}._linear_layers.{i}.weight"]
                l.bias.data = sd[f"{prefix}._linear_layers.{i}.bias"]
            return layers

        self.lab = ff("label_classifier", H, [H, 3])
        self.rat = ff("rationale_classifier", 2 * H, [H, 1])
        self.enc = enc.to(dev()).eval()
        self.lab.to(dev()), self.rat.to(dev())
        self.torch = torch
        log(f"[MultiVerS {ckpt_name}] label_threshold={self.label_threshold} "
            f"rationale_threshold={self.rationale_threshold}")

    @staticmethod
    def _ff(layers, x):
        import torch.nn.functional as F
        x = F.gelu(layers[0](x))  # activations: [GELU, Identity]; dropout off at eval
        return layers[1](x)

    def predict(self, claim, title, sentences):
        torch, tok = self.torch, self.tok
        eos = tok.eos_token
        text = claim + eos + title + eos + eos.join(sentences)
        ids = tok(text, truncation=True, max_length=4096)["input_ids"]
        t = torch.tensor([ids], device=dev())
        is_eos = t[0] == tok.eos_token_id
        first_eos = torch.where(is_eos)[0][0]
        gam = ((t[0] == tok.bos_token_id) | is_eos | (torch.arange(len(ids), device=dev()) < first_eos)).long()
        sent_idx = torch.where(is_eos)[0][2:]
        with torch.no_grad():
            o = self.enc(input_ids=t, attention_mask=torch.ones_like(t), global_attention_mask=gam[None])
            pooled = o.pooler_output
            lp = torch.softmax(self._ff(self.lab, pooled), dim=1)[0]
            if self.label_threshold is None:
                li = int(lp.argmax())
            else:
                lt = lp.clone()
                lt[1] = self.label_threshold
                li = int(lt.argmax())
            ss = o.last_hidden_state[0, sent_idx]
            rp = torch.sigmoid(self._ff(self.rat, torch.cat([pooled.expand_as(ss), ss], dim=1)).squeeze(-1))
            rat = [int(i) for i in torch.where(rp >= self.rationale_threshold)[0].cpu()]
        return self.LABELS[li], rat

    def close(self):
        del self.enc
        self.torch.cuda.empty_cache()


# ----------------------------- driver -----------------------------
def evaluate_preds(claims, oracle_pairs, gold_oracle, open_pairs, pv):
    orc, opn = defaultdict(dict), defaultdict(dict)
    for c, d in oracle_pairs:
        orc[c][d] = pv[(c, d)]
    for c, d in open_pairs:
        opn[c][d] = pv[(c, d)]
    return {"oracle_3class": three_class(gold_oracle, [pv[p][0] for p in oracle_pairs]),
            "oracle_abstract_level": scifact_abstract_metrics(claims, orc),
            "open_abstract_level": scifact_abstract_metrics(claims, opn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-from", default="claude-haiku-5.5",
                    help="whose full-pipeline top-3 is the shared open-setting input")
    ap.add_argument("--llms", default="anthropic/claude-haiku-5.5,deepseek/deepseek-v4.1-flash-1017,"
                                      "anthropic/claude-sonnet-5.5")
    ap.add_argument("--skip", default="", help="comma list of verifiers to skip")
    args = ap.parse_args()
    skip = set(args.skip.split(","))

    sci = {}
    for line in open(f"{DATA}/scifact_raw/data/corpus.jsonl"):
        r = json.loads(line)
        sci[r["doc_id"]] = r
    claims = {str(json.loads(l)["id"]): json.loads(l) for l in open(f"{DATA}/scifact_raw/data/claims_dev.jsonl")}
    oracle_pairs = [(c, d) for c, cl in claims.items() for d in cl["cited_doc_ids"]]
    gold_oracle = [(claims[c]["evidence"].get(str(d)) or [{"label": "NEI"}])[0]["label"] for c, d in oracle_pairs]
    vin = json.load(open(f"{DATA}/verdict_inputs.json"))[args.open_from]
    open_pairs = [(c, int(d)) for c in claims for d in vin[c]]
    all_pairs = sorted(set(oracle_pairs) | set(open_pairs))
    log(f"pairs: oracle={len(oracle_pairs)} open={len(open_pairs)} union={len(all_pairs)}")

    pred_path = f"{DATA}/verdict_baseline_preds.json"
    store = json.load(open(pred_path)) if os.path.exists(pred_path) else {}
    results = {}

    neural = [("VeriSci", lambda: VeriSci()),
              ("MultiVerS-zs (fever_sci)", lambda: MultiVerS("fever_sci")),
              ("MultiVerS-sf* (scifact, CONTAMINATED)", lambda: MultiVerS("scifact"))]
    for name, mk in neural:
        if name.split()[0] in skip:
            continue
        have = store.setdefault(name, {})
        missing = [(c, d) for c, d in all_pairs if f"{c}|{d}" not in have]  # only new pairs are predicted
        if missing:
            log(f"[{name}] predicting {len(missing)} new pairs ...")
            m = mk()
            for c, d in missing:
                have[f"{c}|{d}"] = m.predict(claims[c]["claim"], sci[d]["title"], sci[d]["abstract"])
            if hasattr(m, "close"):
                m.close()
            del m
            json.dump(store, open(pred_path, "w"))
        pv = {(c, d): tuple(store[name][f"{c}|{d}"]) for c, d in all_pairs}
        results[name] = evaluate_preds(claims, oracle_pairs, gold_oracle, open_pairs, pv)
        log(f"[{name}] {json.dumps(results[name]['oracle_3class'])}")

    for model in [m for m in args.llms.split(",") if m]:
        llm = LLM(model, reasoning=None if model in REASONING_MANDATORY else False)
        pv, n_fail = run_verdict(llm, claims, sci, all_pairs)
        tag = "LLM " + model.split("/")[-1]
        results[tag] = evaluate_preds(claims, oracle_pairs, gold_oracle, open_pairs, pv)
        results[tag]["parse_failures"] = n_fail
        log(f"[{tag}] new calls={llm.n_calls} cost=${llm.cost:.3f}")

    out = {"dataset": "SciFact dev (= BEIR test claims)", "open_input": f"top-3 of full pipeline [{args.open_from}]",
           "n_oracle_pairs": len(oracle_pairs), "n_open_pairs": len(open_pairs), "results": results,
           "notes": {"MultiVerS-sf*": "trained on a file that contains all 300 evaluation claims; not comparable",
                     "VeriSci": "trained on SciFact train; dev used for checkpoint selection in the original paper"}}
    json.dump(out, open(f"{HERE}/results_baselines_verdict.json", "w"), indent=2)
    for k, v in results.items():
        o, a, p = v["oracle_3class"], v["oracle_abstract_level"], v["open_abstract_level"]
        print(f"{k:42s} oracle acc={o['accuracy']:.3f} mF1={o['macro_f1']:.3f} | oracle AL label F1="
              f"{a['label_only']['F1']:.3f} rat F1={a['rationale_selected']['F1']:.3f} | open label F1="
              f"{p['label_only']['F1']:.3f} (P {p['label_only']['P']:.3f} R {p['label_only']['R']:.3f}) "
              f"rat F1={p['rationale_selected']['F1']:.3f}")


if __name__ == "__main__":
    main()
