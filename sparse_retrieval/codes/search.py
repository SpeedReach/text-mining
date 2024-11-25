from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import os


def search(searcher, query, args):
    output = open(args.output, 'w')
        
    print(f'Do {args.method} search...')
    for qid, qtext in tqdm(query.items()):
        hits = searcher.search(qtext, k=args.k)
        for i in range(len(hits)):
            # trec format: qid Q0 docid rank score method
            output.write(f'{qid} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} {args.method}\n')
