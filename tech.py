from __future__ import division
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from nltk.util import ngrams
from gensim import corpora, models, similarities
import logging
from gensim.models import *
from nltk.corpus import wordnet as wn
from itertools import product
from math import log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from pylab import *
from scipy.sparse import *
from sklearn.cluster import KMeans
from clustering import *

class tfidf:
	def find_ngrams(self,input_list, n):
		return zip(*[input_list[i:] for i in range(n)])
	def cosine(self,a,b,N,mapping):
		x,y,ans=0,0,0
		for i in mapping.values():
			print i
			ans=ans+(i[a]*i[b])
			x =x+ i[a]**2
			y=y+i[b]**2
		try:
			return ans/pow(x*y,0.5)
		except:
			return 0
	def _tfidf(self,text,text2):

		matrix0 = [[-1 for i in range(len(text))] for j in range(len(text))]
		matrix1 = [[-1 for i in range(len(text))] for j in range(len(text))]
	  	matrix2 = [[-1 for i in range(len(text))] for j in range(len(text))]
	  	matrix_final= [[-1 for i in range(len(text))] for j in range(len(text))]
	  	lis=[{},{},{}]
		mapping = [{},{},{}]
		N=len(text)
		# initinalize mapping and list
		# print "here"
		for i in range(len(text)):
			for j in range(1,4):

				for k in self.find_ngrams(text[i],j):
					lis[j-1][' '.join(k)]=[]
					mapping[j-1][' '.join(k)]=[0 for jj in range(N)]
		for i in range(len(text)):
			for j in range(1,4):
				for k in self.find_ngrams(text[i],j):
					lis[j-1][' '.join(k)].append(i)
		for i in range(len(text)):
			# print text[i 	]
			for j in range(1,4):
				# print j
				b = self.find_ngrams(text[i],j)
				for k in b:
					try:
						tf = b.count(k)
			  			idf = log(N/len(lis[j-1][' '.join(k)]))
				  		score= tf*idf
				  		mapping[j-1][' '.join(k)][i]=score
				  	except:             
				  		pass
		print "here----------------------------------------------------------------------"
		# print matrix0
		print len(matrix0)
		for i in range(len(text)):
			print i
			for j in range(len(text)):
				if i==j:continue
				matrix0[i][j]=self.cosine(i,j,N,mapping[0])     
		  		matrix1[i][j]=self.cosine(i,j, N,mapping[1])
		# Check Point print "Here"
		for i in range(len(text)):
			for j in range(len(text)):
				matrix_final[i][j]=10000*((1/6)*matrix0[i][j] + (1/2)*matrix1[i][j] )#+ (1/2)*matrix2[i][j])
		print "here"
		c = clustering()
		return c.clustering(text,14,matrix_final,text2)
		# return 
