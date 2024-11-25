python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/collection \
  --index indexes/collection \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
