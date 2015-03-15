from numpy import zeros
from scipy.linalg import svd
#following needed for TFIDF
from math import log
from numpy import asarray, sum
from jaccard import *
from tf import *
from tech import *
from similarity import *
from nltk.corpus import stopwords

# 
import numpy as np
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict
# 


titles=[]
title=[]
d=0
for i in range(1,100):
    f1=open("data/"+str(i),'r')
    # d=0
    for j in f1.readlines():
        title.append(j[:150])
        # print title[-1]
        # if "Incremental" in title[-1]:
        #     print d
        #     raw_input()
        break
    f1.close()
    f=open("summaries/part 1/summary_"+str(i),'r')
    temp=""
    for j in f.readlines():
        temp+=j
    f.close()
    titles.append(temp)
    d=d+1

# titles = ["The Neatest Little Guide to Stock Market Investing",
#             # "The Neatest Little Guide to Stock Market Investing",
#             # "The Neatest Little Guide to Stock Market Investing",
#             # "The Neatest Little Guide to Stock Market Investing",
#             # "The Neatest Little Guide to Stock Market Investing",
#           "Investing For Dummies, 4th Edition",
#           "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
#           "The Little Book of Value Investing",
#           "Value Investing: From Graham to Buffett and Beyond",
#           "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
#           "Investing in Real Estate, 5th Edition",
#           "Stock Investing For Dummies",
#           "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
#           ]
# stopwords = ['and','edition','for','in','little','of','the','to']
stopwords = set(stopwords.words('english') + list(punctuation))
ignorechars = ''',:'!'''

class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0     
    # 
    # 
    def cosine_sim(self,u,v):
        return np.dot(u,v) / (sqrt(np.dot(u,u)) * sqrt(np.dot(v,v)))   
    def corpus2vectors(self,corpus):
        def vectorize(sentence, vocab):
            return [sentence.split().count(i) for i in vocab]
        vectorized_corpus = []
        vocab = sorted(set(chain(*[i.lower().split() for i in corpus])))
        for i in corpus:
            vectorized_corpus.append((i, vectorize(i, vocab)))
        return vectorized_corpus, vocab

    # 
    # 
    def parse(self, doc):
        words = doc.split();
        for w in words:
            w = w.lower().translate(None, self.ignorechars)
            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1      
    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i,d] += 1
    def calc(self):
        print "here"
        self.U, self.S, self.Vt = svd(self.A)
    def TFIDF(self):
        WordsPerDoc = sum(self.A, axis=0)        
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
    def printA(self):
        print 'Here is the count matrix'
        print self.A
    def printSVD(self):
        print 'Here are the singular values'
        print self.S
        print 'Here are the first 3 columns of the U matrix'
        print -1*self.U[:, 0:3]
        print 'Here are the first 3 rows of the Vt matrix'
        print -1*self.Vt
    def reco(self):
        def read(file_):
            f=open(file_,'r')
            sent=""
            for i in f.readlines():
                sent+=i
            return sent
        l = len(self.Vt[0])
        L= len(self.Vt)
        print L,l
        dic={}
        for i in range(L):
            dic[i]=[]
        for i in range(L):
            for j in range(i+1,L):
                p=0
                for k in range(1,l):
                    p=p+pow(self.Vt[k][i]-self.Vt[k][j],2)
                dic[i].append((int(10000000*p/l),j))
                dic[j].append((int(10000000*p/l),i))
                dic[i].sort()
                dic[j].sort()
        # print dic[index]
        for _ in range(100):
            print "Enter the document number to get get recommendation for "
            index=input()
            print "Enter Number Of results you want "
            y=input()
            print "******************Getting Recommendation for \"" +str(title[index][:150].upper())+"\" "
            # print titles[index]
            print ""
            c=1
            print "Rank                                          Title                                                                                               Cosine Similarity"
            print "------------------------------------------------------------------------------------------------------------------------------------------------------------------"
            # counter=0
            for i in dic[index][::-1]:
                all_sents=[read("summaries/part 1/summary_"+str(i[1])),read("summaries/part 1/summary_"+str(index+1))]
                corpus, vocab = self.corpus2vectors(all_sents)
                # print dic[index][::-1]
                print str(c)+". ",
                print title[i[1]][:120]+"...",
                print i[1],
                print "        ",self.cosine_sim(corpus[0][1],corpus[1][1])
                c=c+1
                if c==y+1:break



mylsa = LSA(stopwords, ignorechars)
for t in titles:
    mylsa.parse(t)
mylsa.build()
# mylsa.printA()
mylsa.TFIDF()
# mylsa.printA()
mylsa.calc()
# mylsa.printSVD()

mylsa.reco()