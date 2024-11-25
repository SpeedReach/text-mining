import numpy as np
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher
from typing import Dict, List, Tuple, Set

class SimpleHit:
    def __init__(self, docid, score):
        self.docid = docid
        self.score = score


class BM25Searcher:
    def __init__(self, index_path):
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=2, b=0.75)
    
    def search(self, query, k=1000):
        return self.searcher.search(query, k=k)
    from typing import Dict, List, Tuple, Set


class LaplaceLanguageModel:
    def __init__(self, index_path: str):
        """Initialize with collection statistics"""
        self.searcher = LuceneSearcher(index_path)
        self.index_reader = LuceneIndexReader(index_path)
        
        # Collection statistics
        stats = self.index_reader.stats()
        self.total_terms = stats['total_terms']        # t: total terms in corpus
        self.vocab_size = stats['unique_terms']        # k: vocabulary size
        
    def _get_background_prob(self, term: str) -> float:
        """Calculate P(w|C) = cf/|C|"""
        term_info = self.index_reader.get_term_counts(term, analyzer=None)
        cf = term_info[1] if term_info else 0  # collection frequency
        return cf / self.total_terms if cf > 0 else 0
    
    def _score_document(self, query_terms: List[str], docid: str) -> float:
        """
        Score using query likelihood with Laplace smoothing:
        log P(Q|D) = Σ log P(w|D)
        where P(w|D) = (tf + 1)/(doclen + V) 
        
        This implements proper Laplace smoothing where:
        - Add 1 to term frequencies
        - Add V (vocabulary size) to denominator
        """
        # Get document vector and length
        doc_vector = self.index_reader.get_document_vector(docid)
        if not doc_vector:
            return float('-inf')
        
        doc_length = sum(doc_vector.values())
        
        # Score query terms
        score = 0.0
        
        # Using proper Laplace smoothing
        for term in query_terms:
            # Get term frequency in document
            tf = doc_vector.get(term, 0)
            
            # Apply Laplace smoothing formula:
            # P(w|D) = (tf + 1)/(doclen + V)
            p_w_d = (tf + 1) / (doc_length + self.vocab_size)
            
            # Add log probability
            score += np.log(p_w_d)
            
        return score

    def search(self, query: str, k: int = 1000) -> List[SimpleHit]:
        """Search using query likelihood model"""
        # Process query
        query_terms = query.lower().split()
        
        # Get candidates
        candidates = self.searcher.search(query, k=k)
        
        # Score documents
        scores = []
        for hit in candidates:
            score = self._score_document(query_terms, hit.docid)
            if score != float('-inf'):
                scores.append((hit.docid, score))

        # Sort by score and take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        
        return [SimpleHit(docid=docid, score=score) for docid, score in scores]


class JelinekMercerLanguageModel:
    def __init__(self, index_path: str, lambda_param: float = 0.2):
        """
        Initialize language model with Jelinek-Mercer smoothing
        
        Args:
            index_path: Path to Lucene index
            lambda_param: Weight for document model (1-lambda is weight for collection model)
        """
        self.searcher = LuceneSearcher(index_path)
        self.index_reader = LuceneIndexReader(index_path)
        
        # Collection statistics
        stats = self.index_reader.stats()
        self.total_terms = stats['total_terms']  # Total terms in corpus
        
        # Jelinek-Mercer parameter
        self.lambda_param = lambda_param
        
        # Cache for background probabilities
        self._background_prob_cache: Dict[str, float] = {}
        
    def _get_background_prob(self, term: str) -> float:
        """Calculate P(w|C) = cf/|C|"""
        if term not in self._background_prob_cache:
            term_info = self.index_reader.get_term_counts(term, analyzer=None)
            cf = term_info[1] if term_info else 0  # collection frequency
            self._background_prob_cache[term] = cf / self.total_terms
        return self._background_prob_cache[term]
    
    def _score_document(self, query_terms: List[str], docid: str) -> float:
        """
        Score using query likelihood with Jelinek-Mercer smoothing:
        P(w|d) = λP(w|D) + (1-λ)P(w|C)
        where:
        - P(w|D) = tf/doclen (maximum likelihood)
        - P(w|C) = cf/total_terms (background model)
        - λ = weight parameter
        """
        # Get document vector
        doc_vector = self.index_reader.get_document_vector(docid)
        if not doc_vector:
            return float('-inf')
        
        # Get document length
        doc_length = sum(doc_vector.values())
        if doc_length == 0:
            return float('-inf')
        
        # Initialize log score
        log_score = 0.0
        
        # Score each query term
        for term in query_terms:
            # Get term frequency in document
            tf = doc_vector.get(term, 0)
            
            # Calculate maximum likelihood P(w|D) = tf/doclen
            p_w_d = tf / doc_length
            
            # Get background probability P(w|C)
            p_w_c = self._get_background_prob(term)
            
            # Apply Jelinek-Mercer smoothing
            # p(w) = λP(w|D) + (1-λ)P(w|C)
            p_w = (self.lambda_param * p_w_d) + ((1 - self.lambda_param) * p_w_c)
            
            # Add log probability
            if p_w > 0:
                log_score += np.log(p_w)
            else:
                return float('-inf')
        
        return log_score

    def search(self, query: str, k: int = 1000) -> List[SimpleHit]:
        """Search using query likelihood with Jelinek-Mercer smoothing"""
        # Clear background probability cache
        self._background_prob_cache.clear()
        
        # Process query
        query_terms = query.lower().split()
        
        # Get initial candidates (using more for better recall)
        candidates = self.searcher.search(query, k=k*5)
        
        # Score documents
        scores = []
        for hit in candidates:
            score = self._score_document(query_terms, hit.docid())
            if score != float('-inf'):
                scores.append((hit.docid(), score))
        
        # Sort by score and take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        
        return [SimpleHit(docid=docid, score=score) for docid, score in scores]
