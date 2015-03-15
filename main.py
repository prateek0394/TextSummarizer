from jaccard import *
from tf import *
from tech import *
from similarity import *
from nltk.corpus import stopwords
import copy
def main():
  sp = set(stopwords.words('english') + list(punctuation))
  # print punctuation
  # input()
  for paper in range(1):
	  print paper
	  f1=open("test.txt","r")
	  text=""
	  for i in f1.readlines():
	    text=text+i
	  # initials=text[:200]
	  # print initials
	  text =  text.split('.')[:-2]  
	  text2=copy.copy(text)
	  # print text2[:5]
	  for i in range(len(text)):
	      b=[]
	      text[i]=text[i].split()
	      for j in text[i]:
	        if j not in sp:
	          b.append(j)
	      text[i]=b        
	  # j = similar()
	  # final=j.s(text,text2)
	  # print text
	  #j = Jaccard()
	  #final=j.jc(text,text2)
	  print text
	  j = tfidf()
	  final = j._tfidf(text[4:-6],text2)
	  print "here"
	  pr = "summary_"+str(paper)
	  f=open(pr,"w")
	  # f.write(initials)
	  for i in final:
	  	print i
	  	f.write(i)
	  f.close()
	  f1.close()
	  	# f.write()

	  # j=similar()
	  # j.s(text[5:])


if __name__ == '__main__':
  main()
