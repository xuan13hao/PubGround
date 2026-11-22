"""
PubMed Sentence Matcher - High Accuracy Multi-Stage Matching System

This module implements a sophisticated sentence matching system that finds the most
relevant sentences in PubMed articles for a given input sentence, using:
1. LLM-powered query optimization
2. Multi-strategy PubMed search
3. Full-text retrieval and intelligent sentence splitting
4. Multi-signal similarity calculation (semantic + keyword + structure + context)
5. MedCPT cross-encoder reranking
6. Final ranking and deduplication
"""

import asyncio
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple, Any
import xml.etree.ElementTree as ET

try:
    from Bio import Entrez
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

from .llm_utils import create_chat_completion
from .embeddings import Memory
from .config import Config
from .pubmed_citation import PubMedCitationEnhancer
from .reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL
from .retrieval import LitSenseRetriever, LocalCorpus, doc_abstract_text, doc_text, rrf_fuse, RRF_K
from .verdict import VerdictJudge, LLMCall


def _parse_query_list(response: Optional[str]) -> List[str]:
    """JSON array of query strings; from a truncated array keep every complete "..." element."""
    if not response:
        return []
    t = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", response.strip()))
    try:
        qs = json.loads(t)
    except Exception:
        m = re.search(r"\[.*\]", t, re.S)
        try:
            qs = json.loads(m.group(0)) if m else None
        except Exception:
            qs = None
        if qs is None:
            qs = [json.loads(f'"{x}"') for x in re.findall(r'"((?:[^"\\]|\\.)*)"\s*[,\]]', t)]
    if isinstance(qs, list) and qs and all(isinstance(q, str) for q in qs):
        return qs
    return []


class PubMedSentenceMatcher:
    """
    High-accuracy sentence matcher for PubMed articles.
    
    Uses multi-stage approach with LLM enhancement and multi-signal similarity.
    """
    
    def __init__(
        self,
        email: str = "hrteam@h2-alpha.com",
        tool: str = "GPT-Researcher",
        config: Optional[Config] = None,
        similarity_threshold: float = 0.5,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.25,
        structure_weight: float = 0.15,
        context_weight: float = 0.1,
        use_reranker: bool = True,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        rerank_candidates: int = 50,
        reranker_device: Optional[str] = None,
        dense_recall: Optional[str] = "litsense",
        local_corpus: Optional[LocalCorpus] = None,
        rrf_k: int = RRF_K,
        llm_call: Optional[LLMCall] = None,
    ):
        """
        Initialize the PubMed sentence matcher.

        Args:
            email: Email for PubMed API identification
            tool: Tool name for PubMed API
            config: Config object (if None, creates new one)
            similarity_threshold: Minimum similarity score to consider
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword overlap (0-1)
            structure_weight: Weight for structure similarity (0-1)
            context_weight: Weight for context relevance (0-1)
            use_reranker: Enable cross-encoder reranking stage (highest-leverage
                precision improvement). Falls back gracefully to the multi-signal
                score if torch/transformers or the model are unavailable.
            reranker_model: HuggingFace model id for the cross-encoder reranker.
            rerank_candidates: Max candidates (by first-pass score) sent to the
                reranker. Cross-encoders are expensive, so we bound the top-N.
            reranker_device: Force 'cuda' / 'cpu' (None = auto-detect).
            dense_recall: Second first-stage retriever fused with E-utilities by RRF when
                searching PubMed: "litsense" (NCBI LitSense 2.0, MedCPT dense search over
                PubMed) or None for keyword search only.
            local_corpus: Search this LocalCorpus (BM25 + MedCPT dense) instead of PubMed.
            rrf_k: Reciprocal Rank Fusion constant.
            llm_call: Optional async (messages, temperature, max_tokens) -> str used for
                every LLM call (query expansion, verification); defaults to Config's LLM.
        """
        self.email = email
        self.tool = tool
        self.config = config or Config()
        self.similarity_threshold = similarity_threshold

        # Cross-encoder reranker (lazy-loaded on first use)
        self.use_reranker = use_reranker
        self.rerank_candidates = rerank_candidates
        self.reranker = (
            CrossEncoderReranker(model_name=reranker_model, device=reranker_device)
            if use_reranker
            else None
        )
        
        # Validate weights sum to 1.0
        total_weight = semantic_weight + keyword_weight + structure_weight + context_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.weights = {
            'semantic': semantic_weight,
            'keyword': keyword_weight,
            'structure': structure_weight,
            'context': context_weight
        }
        
        # Initialize PubMed enhancer
        self.pubmed_enhancer = PubMedCitationEnhancer(email=email, tool=tool)
        
        # First-stage retrieval: E-utilities (+ LitSense dense recall) or a local corpus, fused by RRF
        self.local_corpus = local_corpus
        self.dense = LitSenseRetriever() if (dense_recall == "litsense" and local_corpus is None) else None
        if dense_recall not in (None, "litsense"):
            raise ValueError(f"dense_recall must be 'litsense' or None, got {dense_recall!r}")
        self.rrf_k = rrf_k
        self.llm_call = llm_call

        # Embedding model for the Stage 4 shortlist, created on first use
        self._embeddings = None
        
        # Initialize sentence tokenizer
        self._init_sentence_tokenizer()
        
        # Cache for embeddings and API calls
        self._embedding_cache = {}
        self._article_cache = {}
    
    def _init_sentence_tokenizer(self):
        """Initialize sentence tokenizer (nltk or fallback)."""
        if NLTK_AVAILABLE:
            try:
                from nltk.tokenize import sent_tokenize
                self.sent_tokenize = sent_tokenize
            except:
                self.sent_tokenize = self._simple_sentence_split
        else:
            self.sent_tokenize = self._simple_sentence_split
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            self.memory = Memory(self.config.embedding_provider, self.config.embedding_model,
                                 **self.config.embedding_kwargs)
            self._embeddings = self.memory.get_embeddings()
        return self._embeddings

    async def _llm(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Single entry point for LLM calls (injectable via ``llm_call``)."""
        if self.llm_call is not None:
            return await self.llm_call(messages, temperature, max_tokens)
        return await create_chat_completion(
            messages=messages, model=self.config.smart_llm_model,
            llm_provider=self.config.smart_llm_provider, temperature=temperature,
            max_tokens=max_tokens, llm_kwargs=self.config.llm_kwargs)

    # ==================== Stage 1: LLM Query Optimization ====================
    
    async def _generate_optimized_queries(self, input_sentence: str) -> List[str]:
        """
        Stage 1: Use LLM to understand sentence intent and generate optimized PubMed queries.
        
        Args:
            input_sentence: The input sentence to match
            
        Returns:
            List of optimized PubMed search queries
        """
        messages = [
            {
                "role": "system",
                "content": "You are a biomedical research expert specializing in PubMed search query optimization."
            },
            {
                "role": "user",
                "content": f"""Given this sentence: "{input_sentence}"

Generate 3-5 optimized PubMed search queries that would best find articles containing similar statements. Consider:
1. Key biomedical concepts and terminology
2. Synonyms and related terms
3. MeSH terms if applicable
4. Different phrasings of the same concept
5. Both broad and specific queries

Respond with ONLY a JSON array of query strings, no other text:
["query1", "query2", "query3"]"""
            }
        ]
        
        try:
            # 1,200 tokens: verbose models write long Boolean/MeSH queries that overflow 500
            response = await self._llm(messages, 0.3, 1200)
            queries = _parse_query_list(response)
            if queries:
                return queries[:5]
            return self._extract_keyword_queries(input_sentence)
        except Exception as e:
            print(f"Error generating optimized queries: {e}")
            # Fallback: extract keywords from sentence
            return self._extract_keyword_queries(input_sentence)
    
    def _extract_keyword_queries(self, sentence: str) -> List[str]:
        """Fallback: extract keywords for search."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', sentence.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        if keywords:
            # Create queries with different combinations
            queries = [
                ' '.join(keywords[:3]),  # First 3 keywords
                ' '.join(keywords),      # All keywords
                ' AND '.join(keywords[:2])  # First 2 with AND
            ]
            return queries
        return [sentence]
    
    # ==================== Stage 2: Multi-Strategy PubMed Search ====================
    
    async def _candidate_rankings(self, claim: str, queries: List[str], k: int = 100) -> List[list]:
        """One best-first ranking per retriever/query (fused later by RRF).

        Local corpus: BM25(claim), BM25(each expanded query), MedCPT dense(claim).
        PubMed: E-utilities relevance ranking per expanded query, a keyword query, and
        LitSense dense recall for the claim (if enabled).
        """
        if self.local_corpus is not None:
            return await asyncio.to_thread(self.local_corpus.rankings, claim, queries, k)
        rankings: List[list] = []
        for query in queries:
            ids = await self._search_pubmed_with_query(query, max_results=20)
            if ids:
                rankings.append(ids)
            time.sleep(0.35)  # Respect rate limits
        keyword_query = ' '.join(self._extract_keywords(claim))
        if keyword_query:
            ids = await self._search_pubmed_with_query(keyword_query, max_results=10)
            if ids:
                rankings.append(ids)
            time.sleep(0.35)
        if self.dense is not None:
            dense = await asyncio.to_thread(self.dense.search, claim, k)
            if dense:
                rankings.append(dense)
        return rankings

    async def _search_pubmed_multi_strategy(
        self,
        input_sentence: str,
        optimized_queries: List[str],
        k: int = 100,
    ) -> List[str]:
        """
        Stage 2: run every retriever and fuse the rankings with Reciprocal Rank Fusion.

        Returns:
            Document ids (PMIDs, or local ids) best-first
        """
        rankings = await self._candidate_rankings(input_sentence, optimized_queries, k)
        return [d for d, _ in rrf_fuse(rankings, self.rrf_k, top_n=k)]

    # ==================== Stage 2.5: Abstract-level reranking ====================

    def _local_abstract(self, doc_id: str) -> Tuple[str, List[str]]:
        doc = self.local_corpus.docs[doc_id]
        a = doc.get("abstract") or ""
        sents = list(a) if isinstance(a, (list, tuple)) else self.sent_tokenize(str(a))
        return str(doc.get("title") or ""), [x for x in sents if str(x).strip()]

    async def _abstract_texts(self, ids: List[str]) -> Dict[str, Tuple[str, List[str]]]:
        """{id: (title, abstract sentences)} for local documents or PubMed (one batched efetch)."""
        if self.local_corpus is not None:
            return {d: self._local_abstract(d) for d in ids if d in self.local_corpus.docs}
        out: Dict[str, Tuple[str, List[str]]] = {}
        for i in range(0, len(ids), 200):
            params = {'db': 'pubmed', 'id': ','.join(ids[i:i + 200]), 'retmode': 'xml',
                      'tool': self.tool, 'email': self.email}
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(params)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    root = ET.fromstring(r.read())
            except Exception as e:
                print(f"Error fetching abstracts: {e}")
                continue
            for art in root.iter('PubmedArticle'):
                pmid = art.findtext('.//MedlineCitation/PMID') or ''
                title = ''.join(art.find('.//ArticleTitle').itertext()) if art.find('.//ArticleTitle') is not None else ''
                abstract = ' '.join(''.join(x.itertext()) for x in art.iter('AbstractText'))
                if pmid and (title or abstract):
                    out[pmid] = (title, [x for x in self.sent_tokenize(abstract) if x.strip()])
            time.sleep(0.35)
        return out

    async def _rerank_abstracts(self, claim: str, fused: List[Tuple[str, float]],
                                texts: Dict[str, Tuple[str, List[str]]]) -> List[Tuple[str, float]]:
        """MedCPT cross-encoder on (claim, title + abstract); falls back to the fused order."""
        cand = [d for d, _ in fused if d in texts]
        if self.reranker is None or not cand or not self.reranker.available:
            return fused
        passages = [(texts[d][0] + " " + " ".join(texts[d][1])).strip() for d in cand]
        scores = await self.reranker.score(claim, passages)
        return sorted(zip(cand, scores), key=lambda x: -x[1])

    async def rank_abstracts(self, claim: str, top_n: int = 100, expand_queries: bool = True,
                             rerank: bool = True) -> List[Tuple[str, float]]:
        """
        PubGround's abstract ranking: Stage 1 query expansion (optional) -> every retriever ->
        RRF fusion -> MedCPT cross-encoder rerank. This is the ranking the sentence pipeline
        starts from, and what benchmark/ evaluates.

        Returns:
            [(doc_id, score)] best-first (rerank logits, or RRF scores if rerank=False)
        """
        queries = await self._generate_optimized_queries(claim) if expand_queries else []
        rankings = await self._candidate_rankings(claim, queries, top_n)
        fused = rrf_fuse(rankings, self.rrf_k, top_n=top_n)
        if not rerank:
            return fused
        texts = await self._abstract_texts([d for d, _ in fused])
        return await self._rerank_abstracts(claim, fused, texts)

    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract meaningful keywords from sentence."""
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'
        }
        
        words = re.findall(r'\b\w+\b', sentence.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        return keywords[:5]  # Top 5 keywords
    
    async def _search_pubmed_with_query(self, query: str, max_results: int = 20) -> List[str]:
        """Search PubMed with a query and return PMIDs."""
        try:
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    retmax=max_results,
                    sort="relevance",
                    retmode="json"
                )
                # When retmode="json", read as text and parse JSON
                json_data = handle.read().decode('utf-8')
                handle.close()
                record = json.loads(json_data)
                return record.get('esearchresult', {}).get('idlist', [])
            else:
                # Fallback to urllib
                params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmax': max_results,
                    'sort': 'relevance',
                    'retmode': 'json',
                    'tool': self.tool,
                    'email': self.email
                }
                url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
                
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode('utf-8')
                    result = json.loads(data)
                    return result.get('esearchresult', {}).get('idlist', [])
        except Exception as e:
            print(f"Error searching PubMed with query '{query}': {e}")
            return []
    
    # ==================== Stage 3: Full-Text Retrieval and Sentence Splitting ====================
    
    async def _fetch_and_parse_articles(self, pmids: List[str]) -> List[Dict]:
        """
        Stage 3: Fetch article content and intelligently split into sentences.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dictionaries with sentences
        """
        if self.local_corpus is not None:
            return [self._local_article(d) for d in pmids if d in self.local_corpus.docs]

        articles = []
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            tasks = [self._fetch_single_article(pmid) for pmid in batch_pmids]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, dict) and result:
                    articles.append(result)
            
            # Rate limiting
            if i + batch_size < len(pmids):
                time.sleep(0.5)
        
        return articles
    
    def _local_article(self, doc_id: str) -> Dict:
        """Article dict for a LocalCorpus document (same shape as the PubMed path)."""
        doc = self.local_corpus.docs[doc_id]
        title, sents = self._local_abstract(doc_id)
        article_data = {'pmid': str(doc.get('pmid', doc_id)), 'title': title,
                        'authors': doc.get('authors', []), 'journal': doc.get('journal', ''),
                        'pub_year': str(doc.get('year', '') or '')}
        abstract = " ".join(sents)
        return {'pmid': doc_id, 'metadata': article_data,
                'sentences': self._intelligent_sentence_split(abstract, article_data),
                'abstract': abstract, 'abstract_sentences': sents, 'full_text': '',
                'title': title, 'authors': article_data['authors'],
                'journal': article_data['journal'], 'year': article_data['pub_year']}

    # ==================== Stage 6 (optional): verdict ====================

    async def _attach_verdicts(self, claim: str, results: List[Dict],
                               abstracts: Dict[str, Tuple[str, List[str]]]) -> None:
        """Label each result's abstract SUPPORT / CONTRADICT / NEI with the benchmarked judge prompt.
        Each abstract is judged once; every result from it gets the same verdict."""
        doc = lambda r: (r.get('article') or {}).get('_doc_id')
        ids = list(dict.fromkeys(doc(r) for r in results if doc(r) in abstracts))
        verdicts = await VerdictJudge(self._llm).judge_many(
            [(claim, abstracts[d][0], abstracts[d][1]) for d in ids])
        by_id = dict(zip(ids, verdicts))
        for r in results:
            r['verdict'] = by_id.get(doc(r))

    async def _fetch_single_article(self, pmid: str) -> Optional[Dict]:
        """Fetch and parse a single article."""
        # Check cache
        if pmid in self._article_cache:
            return self._article_cache[pmid]
        
        try:
            # Fetch article metadata and abstract
            article_data = self.pubmed_enhancer._fetch_article_details(pmid)
            if not article_data:
                return None
            
            # Get abstract text - PubMedCitationEnhancer doesn't extract abstract by default
            # So we fetch it separately
            abstract = await self._fetch_abstract(pmid)
            
            # Try to fetch full text if available (from PMC)
            full_text = await self._fetch_fulltext_if_available(pmid)
            
            # Combine abstract and full text
            combined_text = abstract
            if full_text:
                combined_text += "\n\n" + full_text
            
            if not combined_text.strip():
                return None
            
            # Split into sentences with context
            sentences_with_context = self._intelligent_sentence_split(
                combined_text,
                article_data
            )
            
            article_dict = {
                'pmid': pmid,
                'metadata': article_data,
                'sentences': sentences_with_context,
                'abstract': abstract,
                'full_text': full_text,
                'title': article_data.get('title', ''),
                'authors': article_data.get('authors', []),
                'journal': article_data.get('journal', ''),
                'year': article_data.get('pub_year', ''),
            }
            
            # Cache result
            self._article_cache[pmid] = article_dict
            return article_dict
            
        except Exception as e:
            print(f"Error fetching article {pmid}: {e}")
            return None
    
    async def _fetch_abstract(self, pmid: str) -> str:
        """Fetch abstract from PubMed."""
        try:
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
                xml_data = handle.read().decode('utf-8')
                handle.close()
                
                root = ET.fromstring(xml_data)
                abstract_elem = root.find('.//AbstractText')
                if abstract_elem is not None:
                    return ' '.join(abstract_elem.itertext())
            return ""
        except Exception as e:
            print(f"Error fetching abstract for {pmid}: {e}")
            return ""
    
    async def _fetch_fulltext_if_available(self, pmid: str) -> str:
        """Try to fetch full text from PMC if available."""
        try:
            # First check if article is in PMC
            if ENTREZ_AVAILABLE:
                Entrez.email = self.email
                handle = Entrez.elink(
                    dbfrom="pubmed",
                    db="pmc",
                    id=pmid
                )
                record = Entrez.read(handle)
                handle.close()
                
                if record and record[0].get('LinkSetDb'):
                    pmc_id = record[0]['LinkSetDb'][0]['Link'][0]['Id']
                    # Fetch PMC full text (simplified - would need full PMC parsing)
                    # For now, return empty - can be enhanced later
                    return ""
        except Exception as e:
            pass
        return ""
    
    def _intelligent_sentence_split(self, text: str, article_data: Dict) -> List[Dict]:
        """
        Split text into sentences while preserving context information.
        
        Args:
            text: Text to split
            article_data: Article metadata for context
            
        Returns:
            List of sentence dictionaries with context
        """
        sentences = self.sent_tokenize(text)
        
        sentences_with_context = []
        for i, sentence in enumerate(sentences):
            # Clean sentence
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Get context (previous and next sentences)
            prev_sentence = sentences[i-1].strip() if i > 0 else ""
            next_sentence = sentences[i+1].strip() if i < len(sentences)-1 else ""
            context = f"{prev_sentence} {sentence} {next_sentence}".strip()
            
            sentences_with_context.append({
                'text': sentence,
                'context': context,
                'position': i,
                'article': article_data
            })
        
        return sentences_with_context
    
    # ==================== Stage 4: Multi-Signal Similarity Calculation ====================
    
    async def _calculate_multi_signal_similarity(
        self,
        input_sentence: str,
        candidate_sentences: List[Dict]
    ) -> List[Dict]:
        """
        Stage 4: Calculate similarity using multiple signals and fuse them.
        
        Args:
            input_sentence: Input sentence to match
            candidate_sentences: List of candidate sentences with context
            
        Returns:
            List of matches with similarity scores
        """
        if not candidate_sentences:
            return []
        
        results = []
        
        # Extract all sentence texts for batch processing
        sentence_texts = [c['text'] for c in candidate_sentences]
        
        # Calculate all similarities in parallel where possible
        semantic_scores = await self._batch_semantic_similarity(input_sentence, sentence_texts)
        keyword_scores = [self._keyword_overlap_score(input_sentence, text) for text in sentence_texts]
        structure_scores = [self._sentence_structure_similarity(input_sentence, text) for text in sentence_texts]
        context_scores = await self._batch_context_relevance(input_sentence, candidate_sentences)
        
        # Combine scores
        for i, candidate in enumerate(candidate_sentences):
            final_score = (
                self.weights['semantic'] * semantic_scores[i] +
                self.weights['keyword'] * keyword_scores[i] +
                self.weights['structure'] * structure_scores[i] +
                self.weights['context'] * context_scores[i]
            )
            
            # Only include if above threshold
            if final_score >= self.similarity_threshold:
                results.append({
                    'sentence': candidate['text'],
                    'article': candidate['article'],
                    'context': candidate.get('context', ''),
                    'scores': {
                        'semantic': semantic_scores[i],
                        'keyword': keyword_scores[i],
                        'structure': structure_scores[i],
                        'context': context_scores[i],
                        'final': final_score
                    },
                    'position': candidate.get('position', 0)
                })
        
        return results
    
    async def _batch_semantic_similarity(
        self,
        input_sentence: str,
        candidate_sentences: List[str]
    ) -> List[float]:
        """Calculate semantic similarity using embeddings (batch)."""
        try:
            # Get embeddings
            input_embedding = await self._get_embedding(input_sentence)
            candidate_embeddings = await asyncio.gather(*[
                self._get_embedding(sent) for sent in candidate_sentences
            ])
            
            # Calculate cosine similarity
            scores = []
            for cand_emb in candidate_embeddings:
                if input_embedding and cand_emb:
                    similarity = self._cosine_similarity(input_embedding, cand_emb)
                    scores.append(float(similarity))
                else:
                    scores.append(0.0)
            
            return scores
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return [0.0] * len(candidate_sentences)
    
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (with caching)."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            # Use langchain embeddings
            embedding = await self.embeddings.aembed_query(text)
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except ImportError:
            # Fallback to manual calculation if numpy not available
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
    
    def _keyword_overlap_score(self, sentence1: str, sentence2: str) -> float:
        """Calculate keyword overlap score using TF-IDF-like approach."""
        # Extract keywords from both sentences
        words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
        
        # Remove stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'
        }
        
        words1 = {w for w in words1 if len(w) > 3 and w not in stop_words}
        words2 = {w for w in words2 if len(w) > 3 and w not in stop_words}
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _sentence_structure_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate sentence structure similarity."""
        # Simple approach: compare sentence length and punctuation patterns
        len1, len2 = len(sentence1), len(sentence2)
        len_similarity = 1.0 - abs(len1 - len2) / max(len1, len2, 1)
        
        # Punctuation pattern similarity
        punct1 = re.findall(r'[.!?,;:]', sentence1)
        punct2 = re.findall(r'[.!?,;:]', sentence2)
        punct_similarity = 1.0 if punct1 == punct2 else 0.5
        
        return (len_similarity + punct_similarity) / 2.0
    
    async def _batch_context_relevance(
        self,
        input_sentence: str,
        candidates: List[Dict]
    ) -> List[float]:
        """Calculate context relevance scores (batch)."""
        # For now, use simple keyword matching in context
        # Can be enhanced with embedding-based context matching
        input_keywords = set(re.findall(r'\b\w+\b', input_sentence.lower()))
        input_keywords = {w for w in input_keywords if len(w) > 3}
        
        scores = []
        for candidate in candidates:
            context = candidate.get('context', candidate.get('text', ''))
            context_keywords = set(re.findall(r'\b\w+\b', context.lower()))
            context_keywords = {w for w in context_keywords if len(w) > 3}
            
            if input_keywords and context_keywords:
                overlap = len(input_keywords & context_keywords) / len(input_keywords)
                scores.append(overlap)
            else:
                scores.append(0.0)
        
        return scores
    
    # ==================== Stage 4.5: Cross-Encoder Reranking ====================

    async def _rerank_matches(
        self,
        input_sentence: str,
        matches: List[Dict],
    ) -> List[Dict]:
        """
        Stage 4.5: Rerank first-pass candidates with a cross-encoder.

        The multi-signal score (Stage 4) is a cheap bi-encoder-style first pass
        used only to shortlist candidates. A cross-encoder (MedCPT) then jointly
        scores (query, passage) pairs for fine-grained relevance. This is the
        single largest precision gain in modern retrieval pipelines.

        Adds `scores['rerank']` (normalized) and `scores['rerank_raw']` (logit)
        to each match, and sets `scores['primary']` to the rerank score. If the
        reranker is unavailable, `scores['primary']` falls back to the Stage 4
        `final` score so downstream stages are unaffected.

        Args:
            input_sentence: Original input sentence (query).
            matches: Candidate matches from Stage 4.

        Returns:
            Matches sorted by primary score (descending).
        """
        # Default: primary score == Stage 4 multi-signal score.
        for m in matches:
            m['scores']['primary'] = m['scores']['final']

        if not self.use_reranker or self.reranker is None or not matches:
            return sorted(matches, key=lambda x: x['scores']['primary'], reverse=True)

        # Bound cross-encoder cost: only rerank the top-N by first-pass score.
        shortlisted = sorted(
            matches, key=lambda x: x['scores']['final'], reverse=True
        )[: self.rerank_candidates]

        # Score passage text (with a little context helps the cross-encoder).
        passages = [
            (m.get('context') or m['sentence'])[:1000] for m in shortlisted
        ]
        raw_scores = await self.reranker.score(input_sentence, passages)

        if not raw_scores:  # reranker unavailable -> keep first-pass ordering
            return sorted(matches, key=lambda x: x['scores']['primary'], reverse=True)

        from .reranker import _sigmoid
        for m, raw in zip(shortlisted, raw_scores):
            m['scores']['rerank_raw'] = raw
            m['scores']['rerank'] = _sigmoid(raw)
            m['scores']['primary'] = m['scores']['rerank']

        # Candidates beyond the shortlist keep their first-pass primary score and
        # sort below the reranked ones (they were lower-ranked to begin with).
        return sorted(matches, key=lambda x: x['scores']['primary'], reverse=True)

    # ==================== Stage 5: Final Ranking ====================
    
    def _final_ranking(
        self,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Stage 5: Final ranking considering multiple factors.
        
        Args:
            candidates: Reranked candidate matches
            top_k: Number of top results to return
            
        Returns:
            Final ranked and deduplicated results
        """
        # Calculate article quality scores
        for match in candidates:
            article_quality = self._calculate_article_quality(match['article'])

            # Final rank score = primary relevance * article quality.
            # `primary` is the cross-encoder rerank score when available, else the
            # Stage 4 multi-signal score (set in _rerank_matches).
            primary = match['scores'].get('primary', match['scores']['final'])
            match['final_rank_score'] = primary * article_quality
        
        # Sort by final rank score
        sorted_matches = sorted(
            candidates,
            key=lambda x: x.get('final_rank_score', 0),
            reverse=True
        )
        
        # Deduplicate: keep only best match per article
        deduplicated = self._deduplicate_by_article(sorted_matches)
        
        return deduplicated[:top_k]
    
    def _calculate_article_quality(self, article_data: Dict) -> float:
        """Calculate article quality score based on metadata."""
        score = 1.0
        
        # Prefer articles with more authors (usually indicates collaboration)
        authors = article_data.get('authors', [])
        if len(authors) > 0:
            score *= 1.0 + min(len(authors) / 10.0, 0.2)  # Up to 20% boost
        
        # Prefer recent articles (within last 10 years)
        year = article_data.get('pub_year', '')
        if year:
            try:
                year_int = int(year)
                current_year = 2024
                if current_year - year_int <= 10:
                    score *= 1.1  # 10% boost for recent articles
            except:
                pass
        
        # Prefer articles with journal information
        if article_data.get('journal'):
            score *= 1.05  # 5% boost
        
        return min(score, 1.5)  # Cap at 1.5x
    
    def _deduplicate_by_article(self, matches: List[Dict]) -> List[Dict]:
        """Keep only the best match per article."""
        seen_pmids = {}
        deduplicated = []
        
        for match in matches:
            pmid = match['article'].get('pmid', '')
            if not pmid:
                continue
            
            if pmid not in seen_pmids:
                seen_pmids[pmid] = match
                deduplicated.append(match)
            else:
                # Keep the one with higher score
                if match['final_rank_score'] > seen_pmids[pmid]['final_rank_score']:
                    deduplicated.remove(seen_pmids[pmid])
                    seen_pmids[pmid] = match
                    deduplicated.append(match)
        
        # Re-sort after deduplication
        return sorted(
            deduplicated,
            key=lambda x: x.get('final_rank_score', 0),
            reverse=True
        )
    
    # ==================== Main Function ====================
    
    async def find_matching_sentences(
        self,
        input_sentence: str,
        top_k: int = 10,
        max_articles: int = 50,
        citation_format: str = "mla",
        verify: bool = False,
        candidate_pool: int = 100,
    ) -> List[Dict]:
        """
        Main function: Find top matching sentences in PubMed (or the local corpus).
        
        Args:
            input_sentence: Input sentence to match
            top_k: Number of top results to return
            max_articles: Number of top-ranked abstracts whose sentences are scored
            citation_format: Citation format (mla, apa, nature)
            verify: Also label each result's abstract SUPPORT / CONTRADICT / NEI with the
                benchmarked judge prompt (one LLM call per distinct abstract)
            candidate_pool: Abstracts kept after RRF fusion and reranked by MedCPT
            
        Returns:
            List of matched sentences with citations (and ``verdict`` when verify=True)
        """
        print(f"🔍 Starting sentence matching for: '{input_sentence[:100]}...'")
        
        # Stage 1: LLM Query Optimization
        print("📝 Stage 1: Generating optimized queries...")
        optimized_queries = await self._generate_optimized_queries(input_sentence)
        print(f"   Generated {len(optimized_queries)} queries")
        
        # Stage 2: first-stage retrieval, fused by RRF
        source = "local corpus (BM25 + MedCPT dense)" if self.local_corpus is not None else \
            ("PubMed (E-utilities + LitSense dense)" if self.dense is not None else "PubMed (E-utilities)")
        print(f"🔎 Stage 2: Retrieving from {source}, RRF fusion...")
        rankings = await self._candidate_rankings(input_sentence, optimized_queries, candidate_pool)
        fused = rrf_fuse(rankings, self.rrf_k, top_n=candidate_pool)
        print(f"   {len(rankings)} rankings -> {len(fused)} candidate abstracts")
        
        if not fused:
            print("   No articles found")
            return []

        # Stage 2.5: rank abstracts with the MedCPT cross-encoder (same as rank_abstracts)
        abstracts = await self._abstract_texts([d for d, _ in fused])
        ranked = await self._rerank_abstracts(input_sentence, fused, abstracts)
        pmids = [d for d, _ in ranked][:max_articles]
        print(f"   Kept the top {len(pmids)} abstracts")
        
        # Stage 3: Fetch and Parse Articles
        print("📚 Stage 3: Fetching article content...")
        articles = await self._fetch_and_parse_articles(pmids)
        print(f"   Fetched {len(articles)} articles")
        
        if not articles:
            print("   No article content available")
            return []
        for article in articles:
            article['metadata']['_doc_id'] = article['pmid']
        
        # Extract all candidate sentences
        all_candidates = []
        for article in articles:
            all_candidates.extend(article['sentences'])
        
        print(f"   Extracted {len(all_candidates)} candidate sentences")
        
        # Stage 4: Multi-Signal Similarity
        print("🧮 Stage 4: Calculating similarity scores...")
        matches = await self._calculate_multi_signal_similarity(
            input_sentence,
            all_candidates
        )
        print(f"   Found {len(matches)} matches above threshold")
        
        if not matches:
            print("   No matches found")
            return []

        # Stage 4.5: Cross-Encoder Reranking (MedCPT)
        if self.use_reranker:
            print("🎯 Stage 4.5: Reranking candidates with cross-encoder...")
        matches = await self._rerank_matches(input_sentence, matches)
        if self.use_reranker:
            reranked = sum(1 for m in matches if 'rerank' in m['scores'])
            print(f"   Reranked {reranked} candidates")

        # Top candidates (ordered by rerank score); extra headroom for per-article dedup.
        # An LLM verification stage used to sit here; it was removed because on
        # SciFact it added <=0.007 nDCG@10 over the MedCPT rerank at ~2*top_k LLM
        # calls per query (see benchmark/README.md).
        top_candidates = matches[:top_k * 2]

        # Stage 5: Final Ranking
        print("🏆 Stage 5: Final ranking...")
        final_results = self._final_ranking(top_candidates, top_k)
        
        # Format citations
        for result in final_results:
            article_data = result['article']
            citation = self.pubmed_enhancer.format_mla_citation(article_data, "")
            if citation_format == "apa":
                citation = self.pubmed_enhancer.format_apa_citation(article_data, "")
            elif citation_format == "nature":
                citation = self.pubmed_enhancer.format_nature_citation(article_data, "")
            
            result['citation'] = citation
            result['citation_format'] = citation_format

        # Stage 6 (optional): verdict per abstract
        if verify and final_results:
            print("⚖️  Stage 6: Verifying support (SUPPORT / CONTRADICT / NEI)...")
            await self._attach_verdicts(input_sentence, final_results, abstracts)
        
        print(f"✅ Found {len(final_results)} final matches")
        return final_results

