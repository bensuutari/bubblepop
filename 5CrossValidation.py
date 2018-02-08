import pickle
import numpy as np
from sklearn.metrics import roc_curve,auc
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time
import xgboost as xgb

convecs=pickle.load(open(os.getcwd()+'/models/conservative.wordvecs','rb'))
libvecs=pickle.load(open(os.getcwd()+'/models/liberal.wordvecs','rb'))
np.random.shuffle(convecs)
np.random.shuffle(libvecs)
X=np.concatenate((libvecs,convecs),axis=0)
y=np.concatenate((np.zeros(libvecs.shape[0],),np.ones(convecs.shape[0],)),axis=0)
shuffleind=np.array(range(0,(convecs.shape[0]+libvecs.shape[0])))
np.random.shuffle(shuffleind)
X=X[shuffleind,:]
y=y[shuffleind]

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
classifier=xgb.XGBClassifier(n_estimators=200,max_depth=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
	print('Fitting classifier.')
	startclassifier=time.time()
	probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
	print('Classifier fit complete!  Took: '+str(time.time()-startclassifier))
	# Compute ROC curve and area under the curve
	fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
	tprs.append(interp(mean_fpr, fpr, tpr))
	tprs[-1][0] = 0.0
	roc_auc = auc(fpr, tpr)
	aucs.append(roc_auc)
	plt.plot(fpr, tpr, lw=1, alpha=0.3,
	         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

	i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.tick_params(labelsize=20)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('Receiver operating characteristic',fontsize=20)
plt.legend(loc="lower right",fontsize=20)
plt.show()