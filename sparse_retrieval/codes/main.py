import argparse
from search import *
from util import *
from searcher import BM25Searcher, LaplaceLanguageModel#, JelinekMercerLanguageModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/stemmed", type=str)
    parser.add_argument("--query", default="../data/topics.401.txt", type=str)
    parser.add_argument("--method", default="bm25", type=str,
                        choices=["bm25", "lm_laplace", "lm_jm"])
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--output", default='runs/bm25.run', type=str)
    
    args = parser.parse_args()

    if args.method == "bm25":
        searcher = BM25Searcher(args.index)
    elif args.method == "lm_laplace":
        searcher = LaplaceLanguageModel(args.index)
    else:  # lm_jm
        pass
        searcher = JelinekMercerLanguageModel(args.index)

    query = read_title(args.query)
    search(searcher, query, args)
