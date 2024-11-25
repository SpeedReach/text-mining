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
all: stemmed unstemmed

# Mark collection_jsonl as a phony target since it's not actually a file
.PHONY: collection_jsonl stemmed unstemmed