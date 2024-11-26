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
        return self.searcher.search(query, k=k)[:1000]
    from typing import Dict, List, Tuple, Set

from jnius import autoclass

class JelinekMercerLanguageModel:
    """
    A searcher implementation using Lucene's built-in Jelinek-Mercer language model.
    """
    
    def __init__(self, index_path: str, lambda_param: float = 0.1):
        """
        Initialize the Jelinek-Mercer searcher.
        
        Args:
            index_path (str): Path to the Lucene index
            lambda_param (float): Smoothing parameter lambda (default: 0.1)
                                Should be between 0 and 1
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1")
            
        JLMSimilarity = autoclass('org.apache.lucene.search.similarities.LMJelinekMercerSimilarity')
        
        self.searcher = LuceneSearcher(index_path)
        self.searcher.object.similarity = JLMSimilarity(lambda_param)
    
    def search(self, query: str, k: int = 1000):
        """
        Search the index using the Jelinek-Mercer language model.
        
        Args:
            query (str): Search query
            k (int): Number of results to return (default: 1000)
            
        Returns:
            List of search results
        """
        result = self.searcher.search(query, k=3*k)
        return result[:k]

class LaplaceLanguageModel:
    """
    A searcher implementation using Lucene's built-in Language Model with Laplace/additive smoothing.
    """
    
    def __init__(self, index_path: str, alpha: float = 1.0):
        """
        Initialize the Language Model searcher with Laplace smoothing.
        
        Args:
            index_path (str): Path to the Lucene index
            alpha (float): Laplace smoothing parameter (default: 1.0)
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
            
        # Import required Java classes
        LMDirichletSimilarity = autoclass('org.apache.lucene.search.similarities.LMDirichletSimilarity')
        
        # Initialize searcher with modified Dirichlet similarity
        # We use Dirichlet and set mu=alpha to approximate Laplace smoothing
        self.searcher = LuceneSearcher(index_path)
        self.searcher.object.similarity = LMDirichletSimilarity(alpha)
    
    def search(self, query: str, k: int = 1000):
        """
        Search the index using the language model with Laplace smoothing.
        
        Args:
            query (str): Search query
            k (int): Number of results to return (default: 1000)
            
        Returns:
            List of search results
        """
        # Multiply k by 3 to get more candidates before reranking (common practice)
        result = self.searcher.search(query, k=3*k)
        return result[:k]