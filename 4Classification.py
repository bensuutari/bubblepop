import itertools
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import time

n_classes=2
plotroc=False

def plot_roc_curves(classifiers, X, y):
	"""Produce ROC curves for models given feature set X and labels y
	"""
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=41)
	plt.figure(figsize=(8,8))
	for classifier in classifiers:
		print('Working on '+classifier+' classifier...')
		startclass=time.time()
		clf_train=classifiers[classifier].fit(X_train, y_train)
		pickle.dump(clf_train,open(os.getcwd()+'/classifiers/'+classifier+'.classifier','wb'))
		y_score = clf_train.predict_proba(X_test)[:, 1]

		fpr, tpr, _ = roc_curve(y_test, y_score)
		roc_auc = auc(fpr, tpr)

		plt.plot(fpr, tpr, lw=2, label='%s (area = %0.2f)' % (classifier, roc_auc))
		print(classifier+' classifier took '+str(time.time()-startclass)+' seconds.')
	plt.tick_params(labelsize=20)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate',fontsize = 20)
	plt.ylabel('True Positive Rate',fontsize = 20)
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right",fontsize = 20)
	plt.show()





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=15)
    plt.yticks(tick_marks, classes,fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",fontsize=30,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)


convecs=pickle.load(open(os.getcwd()+'/models/conservative.wordvecs','rb'))
libvecs=pickle.load(open(os.getcwd()+'/models/liberal.wordvecs','rb'))

np.random.shuffle(convecs)
np.random.shuffle(libvecs)

libtest=libvecs[0:1000,:]
libtestlabels=np.zeros((1000,))
libtrain=libvecs[1000:,:]
libtrainlabels=np.zeros((libvecs[1000:,:].shape[0],))

contest=convecs[0:1000,:]
contestlabels=np.ones((1000,))
contrain=convecs[1000:,:]
contrainlabels=np.ones((convecs[1000:,:].shape[0],))

trainvecs=np.concatenate((libtrain,contrain),axis=0)
trainlabels=np.concatenate((libtrainlabels,contrainlabels),axis=0)
testvecs=np.concatenate((libtest,contest),axis=0)
testlabels=np.concatenate((libtestlabels,contestlabels),axis=0)

clf=XGBClassifier(max_depth=5, n_estimators=200)
clf.fit(trainvecs,trainlabels)

y_true=testlabels
y_pred=clf.predict(testvecs)
cnf_matrix=confusion_matrix(y_true, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Conservative','Liberal'], normalize=True,title='')
plt.show()
print(accuracy_score(testlabels, clf.predict(testvecs)))#accuracy_score(y_true, y_pred)
precrecallplot=False
print('Classification complete!')
print('Classification took: '+str(time.time()-startclass)+' seconds.')


if plotroc:
	clfs=dict()
	clfs['SVM']=SVC(probability=True)
	clfs['XGBoost']=XGBClassifier(max_depth=5, n_estimators=200)
	clfs['KNeighbors']=KNeighborsClassifier(n_neighbors=7)
	clfs['RandomForest']=RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_leaf=100, random_state=0)

	X=np.concatenate((libvecs,convecs),axis=0)
	y=np.concatenate((np.zeros(libvecs.shape[0],),np.ones(convecs.shape[0],)),axis=0)
	plot_roc_curves(clfs, X, y)


	









