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
            p_w_d = (tf + 1) / (doc_length + (self.total_terms/self.vocab_size))
            a = (self.total_terms - self.vocab_size)/ self.vocab_size
            b = self._get_background_prob(term)
            c = doc_length + (self.total_terms/ self.vocab_size)
            p_w_d += (a * b / c)
            # Add log probability
            score += np.log(p_w_d)
            
        return score

    def search(self, query: str, k: int = 1000) -> List[SimpleHit]:
        """Search using query likelihood model"""
        # Process query
        query_terms = query.lower().split()
        
        # Get candidates
        candidates = self.searcher.search(query, k=k*5)
        
        # Score documents
        scores = []
        for hit in candidates:
            score = self._score_document(query_terms, hit.docid)
            if score != float('-inf'):
                scores.append((hit.docid, score))
        print(scores)
        # Sort by score and take top k
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        
        return [SimpleHit(docid=docid, score=score) for docid, score in scores]
    


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
            p_w_d = (tf + 1) / (doc_length + (self.total_terms/self.vocab_size))
            a = (self.total_terms - self.vocab_size)/ self.vocab_size
            b = self._get_background_prob(term)
            c = doc_length + (self.total_terms/ self.vocab_size)
            p_w_d += (a * b / c)
            # Add log probability
            score += np.log(p_w_d)
            
        return score

    def search(self, query: str, k: int = 1000) -> List[SimpleHit]:
        """Search using query likelihood model"""
        # Process query
        query_terms = query.lower().split()
        
        # Get candidates
        candidates = self.searcher.search(query, k=k*5)
        
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