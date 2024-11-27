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

runs/bm25-stemmed401.run:
	python3 codes/main.py --method bm25 --output runs/bm25-stemmed401.run --query ../data/topics.401-440.txt --index indexes/stemmed
runs/bm25-unstemmed401.run:
	python3 codes/main.py --method bm25 --output runs/bm25-unstemmed401.run --query ../data/topics.401-440.txt --index indexes/unstemmed

runs/laplace-stemmed401.run:
	python3 codes/main.py --method lm_laplace --output runs/laplace-stemmed401.run --query ../data/topics.401-440.txt --index indexes/stemmed
runs/laplace-unstemmed401.run:
	python3 codes/main.py --method lm_laplace --output runs/laplace-unstemmed401.run --query ../data/topics.401-440.txt --index indexes/unstemmed

runs/jm-stemmed401.run:
	python3 codes/main.py --method lm_jm --output runs/jm-stemmed401.run --query ../data/topics.401-440.txt --index indexes/stemmed
runs/jm-unstemmed401.run:
	python3 codes/main.py --method lm_jm --output runs/jm-unstemmed401.run --query ../data/topics.401-440.txt --index indexes/unstemmed

runs/bm25-stemmed441.run:
	python3 codes/main.py --method bm25 --output runs/bm25-stemmed441.run --query ../data/topics.441-450.txt --index indexes/stemmed
runs/laplace-stemmed441.run:
	python3 codes/main.py --method lm_laplace --output runs/laplace-stemmed441.run --query ../data/topics.441-450.txt --index indexes/stemmed
runs/jm-stemmed441.run:
	python3 codes/main.py --method lm_jm --output runs/jm-stemmed441.run --query ../data/topics.441-450.txt --index indexes/stemmed

runs401: runs/bm25-stemmed401.run runs/bm25-unstemmed401.run runs/laplace-stemmed401.run runs/laplace-unstemmed401.run runs/jm-stemmed401.run runs/jm-unstemmed401.run
runs441: runs/bm25-stemmed441.run runs/laplace-stemmed441.run runs/jm-stemmed441.run
runs: runs401 runs441

learning_to_rank: runs
	python3 codes/build_training_data.py
	python3 codes/train_model.py

score401: 
	perl trec_eval.pl ../data/qrels.401-440.txt runs/bm25-stemmed401.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/bm25-unstemmed401.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/laplace-stemmed401.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/laplace-unstemmed401.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/jm-stemmed401.run
	perl trec_eval.pl ../data/qrels.401-440.txt runs/jm-unstemmed401.run
score441: 
	perl trec_eval.pl ../data/qrels.441-450.txt runs/bm25-stemmed441.run
	perl trec_eval.pl ../data/qrels.441-450.txt runs/laplace-stemmed441.run
	perl trec_eval.pl ../data/qrels.441-450.txt runs/jm-stemmed441.run
	perl trec_eval.pl ../data/qrels.441-450.txt runs/xgboost.run
score: score401 score441


# Mark collection_jsonl as a phony target since it's not actually a file
.PHONY: collection_jsonl stemmed unstemmed all runs401 runs441 runs score401 score441 score