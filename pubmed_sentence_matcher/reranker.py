"""
Cross-encoder reranker for PubMed sentence matching.

Implements a second-stage reranker on top of first-stage (embedding / keyword)
retrieval. This is the single highest-leverage quality improvement in modern
retrieval pipelines: a cross-encoder jointly encodes (query, passage) pairs and
scores fine-grained relevance that bi-encoder cosine similarity misses.

Default model is NCBI's MedCPT-Cross-Encoder, the biomedical-domain SOTA
reranker (PubMedBERT cross-encoder trained on 255M PubMed click logs):
    https://github.com/ncbi/MedCPT

The module lazy-loads torch/transformers so the rest of the package keeps
working (reranker simply disabled) when they are not installed.
"""

import asyncio
import math
import sys
from typing import List, Optional

DEFAULT_RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"


def _log(msg: str) -> None:
    """Log to stderr so we never corrupt a stdio JSON-RPC stream (MCP)."""
    print(f"[reranker] {msg}", file=sys.stderr, flush=True)


class CrossEncoderReranker:
    """
    Lazy-loaded cross-encoder reranker.

    Scores a query against a batch of candidate passages. Higher score = more
    relevant. Raw model outputs are logits; `score()` returns raw logits and
    `score_normalized()` maps them to (0, 1) via a sigmoid for blending with
    other [0, 1] signals.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device
        self._tokenizer = None
        self._model = None
        self._load_failed = False

    @staticmethod
    def dependencies_available() -> bool:
        """True if torch + transformers can be imported."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> bool:
        """Load tokenizer/model on first use. Returns False if unavailable."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            _log(f"torch/transformers not available ({e}); reranker disabled")
            self._load_failed = True
            return False

        try:
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            _log(f"loading {self.model_name} on {self._device} ...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self._device)
            self._model.eval()
            _log("reranker ready")
            return True
        except Exception as e:  # network error, OOM, bad model name, ...
            _log(f"failed to load reranker '{self.model_name}': {e}; disabled")
            self._load_failed = True
            self._model = None
            self._tokenizer = None
            return False

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def _score_sync(self, query: str, passages: List[str]) -> List[float]:
        """Blocking scoring (run inside a thread from async code)."""
        if not passages:
            return []
        if not self._ensure_loaded():
            return [0.0] * len(passages)

        import torch

        scores: List[float] = []
        pairs = [[query, p] for p in passages]
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(dim=1)
            # squeeze collapses a single-item batch to a 0-d tensor
            if logits.dim() == 0:
                scores.append(float(logits.item()))
            else:
                scores.extend(float(x) for x in logits.tolist())
        return scores

    async def score(self, query: str, passages: List[str]) -> List[float]:
        """Return raw relevance logits for each passage (async-safe)."""
        return await asyncio.to_thread(self._score_sync, query, passages)

    async def score_normalized(self, query: str, passages: List[str]) -> List[float]:
        """Return sigmoid-normalized relevance scores in (0, 1)."""
        raw = await self.score(query, passages)
        return [_sigmoid(x) for x in raw]


def _sigmoid(x: float) -> float:
    try:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0
