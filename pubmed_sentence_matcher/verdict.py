"""
Optional verdict stage (``verify=True``): a language-model judge labels each retrieved abstract as
SUPPORT, CONTRADICT or NEI (not enough information) for the claim and cites rationale sentences.

The prompt and parsing here are the ones evaluated in benchmark/ (SciFact, SciFact-Open, HealthVer);
the benchmark imports them from this module.
"""
import asyncio
import json
import re
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

LABELS = ("SUPPORT", "CONTRADICT", "NEI")
TEMPERATURE = 0.0
MAX_TOKENS = 400

# async (messages, temperature, max_tokens) -> response text
LLMCall = Callable[[List[Dict[str, str]], float, int], Awaitable[str]]


def verdict_messages(claim: str, title: str, sentences: Sequence[str]) -> List[Dict[str, str]]:
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


def norm_label(x) -> str:
    x = str(x or "").upper()
    if x.startswith("SUPPORT"):
        return "SUPPORT"
    if x.startswith("CONTRADICT") or x.startswith("REFUTE"):
        return "CONTRADICT"
    return "NEI"


def _parse_json(text: Optional[str]):
    if text is None:
        return None
    t = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", text.strip()))
    try:
        return json.loads(t)
    except Exception:
        pass
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


def parse_verdict(text: Optional[str]) -> Tuple[str, List[int], bool]:
    """-> (label, rationale sentence indices, parsed_ok). Unparseable output counts as NEI."""
    v = _parse_json(text)
    ok = isinstance(v, dict)
    v = v if ok else {}
    rat = v.get("rationale") or []
    rat = [int(i) for i in rat if str(i).lstrip("-").isdigit()] if isinstance(rat, list) else []
    return norm_label(v.get("label")), rat, ok


class VerdictJudge:
    """Labels (claim, abstract) pairs with one LLM call each; identical prompts are sent once."""

    def __init__(self, llm_call: LLMCall, concurrency: int = 8):
        self.llm_call = llm_call
        self.concurrency = concurrency

    async def judge_many(self, items: Sequence[Tuple[str, str, Sequence[str]]]) -> List[Dict]:
        """items: (claim, title, abstract sentences) -> [{label, rationale, rationale_text, parsed}]"""
        msgs = [verdict_messages(c, t, s) for c, t, s in items]
        keys = [json.dumps(m, sort_keys=True) for m in msgs]
        uniq = list(dict.fromkeys(keys))
        sem = asyncio.Semaphore(self.concurrency)

        async def call(k):
            async with sem:
                try:
                    return await self.llm_call(json.loads(k), TEMPERATURE, MAX_TOKENS)
                except Exception:
                    return None

        answers = dict(zip(uniq, await asyncio.gather(*(call(k) for k in uniq))))
        out = []
        for (claim, title, sents), k in zip(items, keys):
            label, rat, ok = parse_verdict(answers[k])
            rat = [i for i in rat if 0 <= i < len(sents)]
            out.append({"label": label, "rationale": rat, "rationale_text": [sents[i] for i in rat],
                        "parsed": ok})
        return out

    async def judge(self, claim: str, title: str, sentences: Sequence[str]) -> Dict:
        return (await self.judge_many([(claim, title, sentences)]))[0]
