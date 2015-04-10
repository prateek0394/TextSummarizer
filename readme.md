Auto Text Summarizer and Recommender System
=======================================

This Text Summarizer is built for summarizing research papers. The Corpus used was of NIPS containing over 1900 documents.
The Recommender system recommends top k documents closely related to the selected document from the corpus. The whole code is in Python. Used python NLTK library. 
The process of summarization is as follows:
    * Initially the each document is processed by removing stop words and stemming. 
    * Using Bag of Word model in summarization.
    * Each document is assumed to be independent. So each document is summarized seperately. In order to summarize,the document is converted into a matrix.This matrix computation is based on three different strategies which are gven weightage based on heuristics. The three strategies are Jaccard value Matrix,tf-idf of n-grams and frequent key-phrase method. 
    * The computed matrix is then used to cluster sentences. K-means clustering algorithm is used to cluster.Semantic similarity of the snetencs is calculated and is used in selection of sentences from the cluster.
    * The selected sentences in order of occurence is gathered. This bag of sentences represent the summary.

--------------------------------------------------------------------------------------------------------

In recommender System, Latent Semantic Analysis is used to calculate the matrix of similarity which order the document from most relevant to least relevant. 

---------------------------------------------------------------------------------------------------------
#### Dependencies
	* sklearn
	* NLTK
	* scipy
	* numpy

---------------------------------------------------------------------------------------------------------


