import os
import pickle
import time
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string


add_stopwords=['permalink','set','default','input','value','video','filter','output','frame','options','stream','new','click','window','share','subscribe','embed','reply','gold','save','advertisement']
stop_words=stopwords.words('english')+add_stopwords+list(string.punctuation)
startwordvecload=time.time()


def clean_text_and_tokenize(doc_collection):
	libdocs=list()
	condocs=list()
	for i,j,k,l,m,n in zip(doc_collection.websitetext,doc_collection.subreddit,doc_collection.submissionID,doc_collection.upvotes,doc_collection.url,doc_collection.title):

		if (type(i) is str):
			if j in ['liberal','democrats','politics','socialism']:
				tokenized_doc=word_tokenize(i)
				cleaneddoc=[w.lower() for w in tokenized_doc if w not in stop_words]
				libdocs.append([FreqDist(cleaneddoc),j,k,l,m,n])
			elif j in ['conservative','republican','libertarian','the_congress']:
				tokenized_doc=word_tokenize(i)
				cleaneddoc=[w.lower() for w in tokenized_doc if w not in stop_words]
				condocs.append([FreqDist(cleaneddoc),j,k,l,m,n])
	return libdocs,condocs

print('Loading reddit data...')
postdata=pickle.load(open('/home/ben/Dropbox/Insight/bubblepop/data/postdata.pickle','rb'))
print('Reddit data loaded!')
print('Start document cleaning...')
startclean=time.time()
libdocs,condocs=clean_text_and_tokenize(postdata)
print('Document cleaning took: '+str(time.time()-startclean)+' seconds.')
print('liberal doc length:'+str(len(libdocs)))
print('conservative doc length:'+str(len(condocs)))

pickle.dump(libdocs,open(os.getcwd()+'/clean_docs/libdocs.pickle','wb'))
pickle.dump(condocs,open(os.getcwd()+'/clean_docs/condocs.pickle','wb'))



