# Directory containing source files
WT2G_DIR := ../data/WT2G
COLLECTION_FILE := data/collection/collection.jsonl
STEMMED_INDEX_DIR := indexes/stemmed
UNSTEMMED_INDEX_DIR := indexes/unstemmed


# Target file that will be created
$(COLLECTION_FILE): $(WT2G_DIR)/*
	python3 codes/convert_wt2g_to_jsonl.py


$(STEMMED_INDEX_DIR): $(COLLECTION_FILE)
	python3 -m pyserini.index.lucene \
		--collection JsonCollection \
		--input data/collection \
		--index $(STEMMED_INDEX_DIR) \
		--generator DefaultLuceneDocumentGenerator \
		--threads 6 \
		--storePositions \
		--storeDocvectors \
		--storeRaw \
		--stemmer porter

# Build unstemmed Lucene index from collection.jsonl
$(UNSTEMMED_INDEX_DIR): $(COLLECTION_FILE)
	python3 -m pyserini.index.lucene \
		--collection JsonCollection \
		--input data/collection \
		--index $(UNSTEMMED_INDEX_DIR) \
		--generator DefaultLuceneDocumentGenerator \
		--threads 6 \
		--storePositions \
		--storeDocvectors \
		--storeRaw \
		--stemmer none

collection_jsonl: $(COLLECTION_FILE)
stemmed: $(STEMMED_INDEX_DIR)
unstemmed: $(UNSTEMMED_INDEX_DIR)
all: stemmed unstemmed runs

runs/bm25-stemmed.run:
	python3 codes/main.py --method bm25 --output runs/bm25-stemmed.run --query ../data/topics.401-440.txt --index indexes/stemmed
runs/bm25-unstemmed.run:
	python3 codes/main.py --method bm25 --output runs/bm25-unstemmed.run --query ../data/topics.401-440.txt --index indexes/unstemmed

runs/laplace-stemmed.run:
	python3 codes/main.py --method lm_laplace --output runs/laplace-stemmed.run --query ../data/topics.401-440.txt --index indexes/stemmed
runs/laplace-unstemmed.run:
	python3 codes/main.py --method lm_laplace --output runs/laplace-unstemmed.run --query ../data/topics.401-440.txt --index indexes/unstemmed

runs/jm-stemmed.run:
	python3 codes/main.py --method lm_jm --output runs/jm-stemmed.run --query ../data/topics.401-440.txt --index indexes/stemmed
runs/jm-unstemmed.run:
	python3 codes/main.py --method lm_jm --output runs/jm-unstemmed.run --query ../data/topics.401-440.txt --index indexes/unstemmed

runs: runs/bm25-stemmed.run runs/bm25-unstemmed.run runs/laplace-stemmed.run runs/laplace-unstemmed.run runs/jm-stemmed.run runs/jm-unstemmed.run

score:
	perl trec_eval.pl ../data/qrels.401-440.txt runs/bm25-stemmed.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/bm25-unstemmed.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/laplace-stemmed.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/laplace-unstemmed.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/jm-stemmed.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/jm-unstemmed.run


# Mark collection_jsonl as a phony target since it's not actually a file
.PHONY: collection_jsonl stemmed unstemmed