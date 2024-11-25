./clean.sh


#############################################################################
# We first convert WT2G files into the jsonl format required by pyserini.   #
# No need this step when Using TrecwebCollection instead of JsonCollection. #
#############################################################################
# python codes/convert_wt2g_to_jsonl.py


##################################################################
# Secondly, we can build index for our WT2G corpus(247491 docs). #
# Use TrecwebCollection to build WT2G corpus(246772).            #
##################################################################
# ./codes/build_index.sh
./codes/build_trecweb_index.sh


##########################################################
# Then, search and store result in the trec_eval format. #
##########################################################
python codes/main.py --query ../data/topics.401.txt --method bm25 --output runs/bm25_1.run


##############################
# Lastly, do the evaluation. #
##############################
echo "BM25 result on 40 queries"
perl trec_eval.pl ../data/qrels.401.txt runs/bm25_1.run
