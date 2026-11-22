"""
First-stage retrieval for PubGround: several ranked candidate lists fused by
Reciprocal Rank Fusion (RRF).

Retrievers
- PubMed (live): NCBI E-utilities keyword search (in PubGroundMatcher) plus
  ``LitSenseRetriever``, NCBI's MedCPT-based dense sentence search over PubMed.
- Local collections (your own abstracts, or benchmark corpora): ``BM25Index``
  (Lucene parameters, as in BEIR/Anserini) and ``MedCPTDenseIndex`` (MedCPT
  query/article encoders), wrapped by ``LocalCorpus``.

torch/transformers/scipy are imported lazily so the rest of the package works
without them.
"""
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

RRF_K = 60
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_FIELD_TAG = re.compile(r"\[[^\]]*\]")
_BOOLEAN = {"and", "or", "not"}

Ranking = List[Tuple[str, float]]


def _log(msg: str) -> None:
    print(f"[retrieval] {msg}", file=sys.stderr, flush=True)


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _TOKEN.findall(text)]


def query_to_bow(query: str) -> List[str]:
    """PubMed query syntax -> bag of words (drops [MeSH]/[tiab] tags and AND/OR/NOT)."""
    return [w for w in tokenize(_FIELD_TAG.sub(" ", query)) if w not in _BOOLEAN]


def rrf_fuse(rankings: Iterable[Sequence], k: int = RRF_K, top_n: Optional[int] = None) -> Ranking:
    """Reciprocal Rank Fusion. Each ranking is a best-first list of ids or (id, score) pairs."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            doc_id = item[0] if isinstance(item, (tuple, list)) else item
            scores[str(doc_id)] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_n] if top_n else fused


# ----------------------------- documents -----------------------------
def doc_title(doc: Dict) -> str:
    return str(doc.get("title") or "")


def doc_abstract_text(doc: Dict) -> str:
    a = doc.get("abstract") or ""
    return " ".join(a) if isinstance(a, (list, tuple)) else str(a)


def doc_text(doc: Dict) -> str:
    return (doc_title(doc) + " " + doc_abstract_text(doc)).strip()


# ----------------------------- BM25 -----------------------------
class BM25Index:
    """Okapi BM25 with Lucene parameters (k1=0.9, b=0.4, idf=log(1+(N-df+.5)/(df+.5)))."""

    def __init__(self, docs: Dict[str, Dict], k1: float = 0.9, b: float = 0.4):
        import numpy as np
        from scipy.sparse import csr_matrix
        self.ids = list(docs)
        vocab: Dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, d in enumerate(self.ids):
            counts: Dict[int, int] = defaultdict(int)
            for w in tokenize(doc_text(docs[d])):
                counts[vocab.setdefault(w, len(vocab))] += 1
            for j, c in counts.items():
                rows.append(i); cols.append(j); vals.append(c)
        tf = csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(self.ids), len(vocab)))
        dl = np.asarray(tf.sum(1)).ravel()
        df = np.bincount(tf.indices, minlength=len(vocab))
        idf = np.log(1 + (len(self.ids) - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / max(dl.mean(), 1e-9))
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self._W = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).tocsc()
        self._vocab = vocab

    def search_tokens(self, tokens: List[str], k: int = 100) -> Ranking:
        import numpy as np
        counts: Dict[int, int] = defaultdict(int)
        for t in tokens:
            if t in self._vocab:
                counts[self._vocab[t]] += 1
        if not counts:
            return []
        s = np.zeros(len(self.ids), dtype=np.float32)
        for j, c in counts.items():
            col = self._W.getcol(j)
            s[col.indices] += c * col.data
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(self.ids[i], float(s[i])) for i in top if s[i] > 0]

    def search(self, query: str, k: int = 100) -> Ranking:
        return self.search_tokens(query_to_bow(query), k)


# ----------------------------- MedCPT dense -----------------------------
class MedCPTDenseIndex:
    """MedCPT bi-encoder: articles as [title, abstract] (article encoder), queries with the query
    encoder; CLS embeddings, dot-product search. Embeddings can be saved/loaded (.pt)."""

    ARTICLE = "ncbi/MedCPT-Article-Encoder"
    QUERY = "ncbi/MedCPT-Query-Encoder"

    def __init__(self, device: Optional[str] = None, batch_size: int = 128, max_length: int = 512):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.ids: List[str] = []
        self.emb = None
        self._q = None

    def _dev(self):
        import torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _encoder(self, name):
        import torch
        from transformers import AutoModel, AutoTokenizer
        dtype = torch.float16 if self._dev() == "cuda" else torch.float32
        return (AutoTokenizer.from_pretrained(name),
                AutoModel.from_pretrained(name, torch_dtype=dtype).to(self._dev()).eval())

    def _embed(self, tok, model, inputs, pairs: bool):
        import torch
        order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i][1] if pairs else inputs[i]))
        out = [None] * len(inputs)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            batch = [inputs[i] for i in idx]
            if pairs:
                enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation=True, padding=True,
                          max_length=self.max_length, return_tensors="pt")
            else:
                enc = tok(batch, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
            with torch.no_grad():
                h = model(**enc.to(self._dev())).last_hidden_state[:, 0].float()
            for j, i in enumerate(idx):
                out[i] = h[j].half().cpu()
        return torch.stack(out)

    def build(self, docs: Dict[str, Dict], log_every: int = 50000) -> "MedCPTDenseIndex":
        import torch
        tok, model = self._encoder(self.ARTICLE)
        self.ids = list(docs)
        parts = []
        for s in range(0, len(self.ids), log_every):
            chunk = self.ids[s:s + log_every]
            parts.append(self._embed(tok, model, [[doc_title(docs[d]), doc_abstract_text(docs[d])] for d in chunk],
                                     pairs=True))
            _log(f"MedCPT article encoder: {min(s + log_every, len(self.ids))}/{len(self.ids)}")
        self.emb = torch.cat(parts)
        del model
        return self

    def save(self, path: str) -> None:
        import torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"ids": self.ids, "emb": self.emb, "model": self.ARTICLE, "max_length": self.max_length}, path)

    @classmethod
    def load(cls, path: str, **kw) -> "MedCPTDenseIndex":
        import torch
        obj = cls(**kw)
        d = torch.load(path)
        obj.ids, obj.emb = list(map(str, d["ids"])), d["emb"]
        return obj

    def encode_queries(self, queries: List[str]):
        if self._q is None:
            self._q = self._encoder(self.QUERY)
        return self._embed(self._q[0], self._q[1], list(queries), pairs=False)

    def search_many(self, queries: List[str], k: int = 100) -> List[Ranking]:
        import torch
        q = self.encode_queries(queries).float().to(self._dev())
        out = []
        for i in range(0, len(q), 64):
            sc = q[i:i + 64] @ self.emb.float().to(self._dev()).T
            v, idx = torch.topk(sc, k=min(k, sc.shape[1]), dim=1)
            for r in range(len(sc)):
                out.append([(self.ids[int(j)], float(x)) for j, x in zip(idx[r], v[r])])
        return out

    def search(self, query: str, k: int = 100) -> Ranking:
        return self.search_many([query], k)[0]


# ----------------------------- LitSense (PubMed dense) -----------------------------
class LitSenseRetriever:
    """NCBI LitSense 2.0: MedCPT dense retrieval of PubMed/PMC sentences. Returns PMIDs ranked by
    their best-matching sentence. Network failures return an empty ranking (E-utilities still runs)."""

    URL = "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/sentences/"

    def __init__(self, timeout: float = 30.0, rerank: bool = True):
        self.timeout = timeout
        self.rerank = rerank

    def search(self, query: str, k: int = 100) -> Ranking:
        params = urllib.parse.urlencode({"query": query, "rerank": str(self.rerank).lower()})
        try:
            with urllib.request.urlopen(f"{self.URL}?{params}", timeout=self.timeout) as r:
                hits = json.loads(r.read().decode("utf-8"))
        except Exception as e:  # network/API problems must not break the pipeline
            _log(f"LitSense unavailable ({e}); continuing without dense recall")
            return []
        best: Dict[str, float] = {}
        for h in hits:
            pmid = str(h.get("pmid") or "")
            if pmid and h.get("score") is not None:
                best[pmid] = max(best.get(pmid, float("-inf")), float(h["score"]))
        return sorted(best.items(), key=lambda x: -x[1])[:k]


# ----------------------------- local collection -----------------------------
class LocalCorpus:
    """A local abstract collection searched by BM25 and/or MedCPT dense retrieval.

    docs: {doc_id: {"title": str, "abstract": str | [sentences], ...optional metadata}}
    """

    def __init__(self, docs: Dict[Union[str, int], Dict], bm25: bool = True,
                 dense: Union[bool, MedCPTDenseIndex, str] = True, device: Optional[str] = None):
        self.docs = {str(k): v for k, v in docs.items()}
        self.bm25 = BM25Index(self.docs) if bm25 else None
        if isinstance(dense, MedCPTDenseIndex):
            self.dense = dense
        elif isinstance(dense, str):          # path to a saved index
            self.dense = MedCPTDenseIndex.load(dense, device=device)
        elif dense:
            self.dense = MedCPTDenseIndex(device=device).build(self.docs)
        else:
            self.dense = None
        if self.dense is not None:
            missing = set(self.docs) - set(self.dense.ids)
            if missing:
                raise ValueError(f"dense index lacks {len(missing)} corpus documents")

    def rankings(self, claim: str, expanded_queries: Sequence[str] = (), k: int = 100) -> List[Ranking]:
        """One ranking per retriever/query: BM25(claim), BM25(each expanded query), dense(claim)."""
        out: List[Ranking] = []
        if self.bm25 is not None:
            out.append(self.bm25.search_tokens(tokenize(claim), k))
            out += [r for r in (self.bm25.search(q, k) for q in expanded_queries) if r]
        if self.dense is not None:
            out.append(self.dense.search(claim, k))
        return [r for r in out if r]
