import gensim
import os
import pickle
import time
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.manifold import TSNE,MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
num_dim=2
show_dimreduc=False
print('Loading pretrained word2vec model...')
model = gensim.models.KeyedVectors.load_word2vec_format(os.getcwd()+'/pretrained_word2vec/GoogleNews-vectors-negative300.bin', binary=True)

def display_reduced_dim(data,dim_reduc_type,liblength,conlength):
	print('Starting '+dim_reduc_type+' embedding...')
	startembed=time.time()
	if dim_reduc_type=='TSNE':
		model_embedded=TSNE(n_components=num_dim).fit_transform(data)
	elif dim_reduc_type=='MDS':
		model_embedded=MDS(n_components=num_dim).fit_transform(data)
	elif dim_reduc_type=='PCA':
		model_embedded=PCA(n_components=num_dim).fit_transform(data)
	else:
		print('Error: dimensionality reduction method not supported (use TSNE, MDS or PCA)')
	print(dim_reduc_type+' embedding took: '+str((time.time()-startembed)/60)+' minutes.')

	#pickle.dump(model_embedded,open(os.getcwd()+'/graphs/low_D_doc2vec/LDAvec_2d_'+dim_reduc_type,'wb'))
	for ii in range(0,model_embedded.shape[0]):
		print('plotted point '+str(ii))
		if ii<liblength:# in ['liberal','democrats','socialism','politics'] 
			plt.plot(model_embedded[ii,0],model_embedded[ii,1],'ob')
		elif ii>=liblength:# in ['conservative','republican','libertarian','the_congress']
			plt.plot(model_embedded[ii,0],model_embedded[ii,1],'or')

	plt.show()

	return model_embedded


def vectorize_docs(docfreqs):
	allvecs=np.zeros((len(docfreqs),300))
	doccounter=0
	totalzeros=0
	for i in docfreqs:
		totalwords=0
		docvector=np.zeros((300,))
		for j in i[0].keys():
			try:
				docvector+=model.wv[j]*(i[0][j])
				totalwords+=i[0][j]
			except:
				pass
		if totalwords>0:
			allvecs[doccounter]=docvector/totalwords
		else:
			totalzeros+=1
			allvecs[doccounter]=docvector
		doccounter+=1
	return allvecs

libdocs=pickle.load(open(os.getcwd()+'/clean_docs/libdocs.pickle','rb'))
condocs=pickle.load(open(os.getcwd()+'/clean_docs/condocs.pickle','rb'))

libvecs=vectorize_docs(libdocs)
convecs=vectorize_docs(condocs)

pickle.dump(libvecs,open(os.getcwd()+'/models/liberal.wordvecs','wb')) 
pickle.dump(convecs,open(os.getcwd()+'/models/conservative.wordvecs','wb')) 

if show_dimreduc:
	display_reduced_dim(np.concatenate((demvecs,convecs),axis=0),'TSNE',demvecs.shape[0],convecs.shape[0])  

