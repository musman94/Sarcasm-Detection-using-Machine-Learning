'''
@rana
bernoulli naive bayes testing code
'''

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import itertools
import pickle
import time
import numpy as np
import scipy as sp
import pandas as pd

start_time = time.time()

feature_file = "feature_vec_test_bnb.p"
label_file = 'label_list_test_bnb.p'
classifier_file = "./bnb_classifier.p"

feature_vec_test = pickle.load(open(feature_file, 'rb'))
print("Got test features")

Y_test = pickle.load(open(label_file, 'rb'))
print("Got test labels")

classifier = pickle.load(open(classifier_file, 'rb'))
print("Got classifier")

print("Testing")
Y_pred = classifier.predict(feature_vec_test)
print("Testing done")

Y_score = classifier.predict_proba(feature_vec_test)
print("Precision-Recall")

average_precision = average_precision_score(Y_test, Y_score[:,1])
print('Average precision-recall score: {0:0.3f}'.format(average_precision))
precision, recall, _ = precision_recall_curve(Y_test, Y_score[:,1])
plt.figure()
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()

print("ROC")

fpr, tpr, _ = roc_curve(Y_test, Y_score[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
lbl = 'ROC curve (area = %0.2f)' % (roc_auc)
plt.plot(fpr, tpr, color='darkorange', lw=lw, label=lbl)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("Confusion matrix")
conf_matrix = confusion_matrix(Y_test, Y_pred)
print(conf_matrix)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len([0,1]))
plt.xticks(tick_marks, [0,1], rotation=45)
plt.yticks(tick_marks, [0,1])
fmt = '.2f'
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center", color="black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


print (classification_report(Y_test, Y_pred))

elapsed_time = time.time() - start_time
print("Elapsed time: %d" %(elapsed_time))
