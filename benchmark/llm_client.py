"""
Minimal OpenRouter (OpenAI-compatible) chat client for the benchmark.

- Key: OPENROUTER_API_KEY, else OPENAI_API_KEY (read from env / benchmark/.env).
- Every response is cached on disk (data/llm_cache/<model>.jsonl) keyed by a hash
  of (model, messages, temperature, max_tokens), so reruns are free and resumable.
- Usage (tokens + USD cost reported by OpenRouter) is appended to data/llm_usage.jsonl.
"""
import hashlib, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REFUSAL = "__CONTENT_FILTER_REFUSAL__"  # cached marker; parse_json() -> None -> caller fallback
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _load_dotenv():
    for p in (os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _key():
    k = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not k:
        raise RuntimeError("Set OPENROUTER_API_KEY (env or benchmark/.env)")
    return k


class LLM:
    # Reasoning is switched off by default: these are short JSON-classification calls
    # and reasoning models otherwise spend the whole max_tokens budget thinking.
    # Some endpoints (e.g. claude-sonnet-5.5) make it mandatory -> pass reasoning=None.
    def __init__(self, model, workers=16, reasoning=False):
        self.model = model
        self.reasoning = reasoning
        self.workers = workers
        os.makedirs(f"{DATA}/llm_cache", exist_ok=True)
        self.cache_path = f"{DATA}/llm_cache/{model.replace('/', '__')}.jsonl"
        self.cache = {}
        if os.path.exists(self.cache_path):
            for line in open(self.cache_path):
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r["content"]
                except Exception:
                    pass
        self._lock = threading.Lock()
        self.cost = 0.0
        self.n_calls = 0

    def _hash(self, messages, temperature, max_tokens):
        s = json.dumps([self.model, messages, temperature, max_tokens], sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def chat(self, messages, temperature=0.0, max_tokens=300, tag=""):
        key = self._hash(messages, temperature, max_tokens)
        if key in self.cache:
            return self.cache[key]
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens,
                "usage": {"include": True}}
        if self.reasoning is False:
            body["reasoning"] = {"enabled": False}
        last = None
        for attempt in range(6):
            try:
                r = requests.post(f"{BASE_URL}/chat/completions",
                                  headers={"Authorization": f"Bearer {_key()}"},
                                  json=body, timeout=120)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                j = r.json()
                if "error" in j:
                    raise RuntimeError(str(j["error"])[:300])
                content = j["choices"][0]["message"].get("content") or ""
                usage = j.get("usage", {}) or {}
                if j["choices"][0].get("finish_reason") == "content_filter":
                    content = REFUSAL  # provider safety filter: deterministic, do not retry
                    break
                if not content.strip():
                    raise RuntimeError("empty content (reasoning ate the token budget?)")
                usage = j.get("usage", {}) or {}
                break
            except Exception as e:  # retry transient errors
                last = e
                time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError(f"{self.model}: giving up after retries: {last}")
        with self._lock:
            self.cache[key] = content
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "content": content}) + "\n")
            with open(f"{DATA}/llm_usage.jsonl", "a") as f:
                f.write(json.dumps({"model": self.model, "tag": tag,
                                    "prompt_tokens": usage.get("prompt_tokens"),
                                    "completion_tokens": usage.get("completion_tokens"),
                                    "cost": usage.get("cost")}) + "\n")
            self.cost += float(usage.get("cost") or 0)
            self.n_calls += 1
        return content

    def map(self, jobs, temperature=0.0, max_tokens=300, tag="", log=None):
        """jobs: list of message lists -> list of responses (same order).
        A job that fails after retries yields None instead of aborting the run."""
        out = [None] * len(jobs)

        def one(i):
            try:
                out[i] = self.chat(jobs[i], temperature, max_tokens, tag)
            except Exception as e:
                if log:
                    log(f"    [{self.model}] job {i} failed: {e}")
            return i

        with ThreadPoolExecutor(self.workers) as ex:
            for n, _ in enumerate(ex.map(one, range(len(jobs)))):
                if log and n % 500 == 0:
                    log(f"    [{self.model}:{tag}] {n}/{len(jobs)}  new-calls={self.n_calls} cost=${self.cost:.3f}")
        return out


def parse_json(text):
    """Lenient JSON extraction (strips ``` fences, grabs the first {...} / [...])."""
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
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


class LocalLLM(LLM):
    """Same interface as LLM, but runs a local HF causal LM (e.g. Qwen3-8B) with batched
    generation. Spec: "local:/abs/path/to/model". Thinking is disabled (enable_thinking=False)
    to match the API models, which run with reasoning off. Responses share the disk cache."""

    def __init__(self, spec, workers=16, reasoning=False, batch_size=48):
        self.path = spec.split(":", 1)[1]
        super().__init__("local__" + os.path.basename(self.path.rstrip("/")), workers=workers,
                         reasoning=reasoning)
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tok = AutoTokenizer.from_pretrained(self.path, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.path, torch_dtype=torch.bfloat16).to("cuda").eval()

    def _prompt(self, messages):
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def chat(self, messages, temperature=0.0, max_tokens=300, tag=""):
        return self.map([messages], temperature, max_tokens, tag)[0]

    def map(self, jobs, temperature=0.0, max_tokens=300, tag="", log=None):
        import torch
        out = [None] * len(jobs)
        keys = [self._hash(m, temperature, max_tokens) for m in jobs]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        for i, k in enumerate(keys):
            if k in self.cache:
                out[i] = self.cache[k]
        if todo:
            self._load()
            # sort by prompt length so each batch pads little
            todo.sort(key=lambda i: len(self._prompt(jobs[i])))
        for b in range(0, len(todo), self.batch_size):
            idx = todo[b:b + self.batch_size]
            enc = self._tok([self._prompt(jobs[i]) for i in idx], return_tensors="pt",
                            padding=True).to("cuda")
            torch.manual_seed(0)
            with torch.no_grad():
                gen = self._model.generate(
                    **enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=None, top_k=None, pad_token_id=self._tok.pad_token_id)
            texts = self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            with open(self.cache_path, "a") as f:
                for i, t in zip(idx, texts):
                    t = t.strip()
                    out[i] = t
                    self.cache[keys[i]] = t
                    f.write(json.dumps({"key": keys[i], "content": t}) + "\n")
            self.n_calls += len(idx)
            if log and (b // self.batch_size) % 10 == 0:
                log(f"    [{self.model}:{tag}] {b + len(idx)}/{len(todo)} generated")
        return out


def make_llm(model, workers=16, reasoning=False):
    """"local:/path" -> LocalLLM (GPU), anything else -> OpenRouter LLM."""
    if model.startswith("local:"):
        return LocalLLM(model, workers=workers, reasoning=reasoning)
    return LLM(model, workers=workers, reasoning=reasoning)
