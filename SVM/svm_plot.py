# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:50:12 2018

@author: abdallah
"""
import matplotlib.pyplot as plt

import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#####
#
## precision    recall    f1-score    support
#class_zero   = [[0.00,0.00,0.00,15139], [0.80,0.00,0.01,15139],
#                [0.58,0.36,0.45,15139], [0.56,0.39,0.46,15139],
#                [0.64,0.65,0.65,15139]]
#class_one    = [[0.50,1.00,0.66,14861], [0.50,1.00,0.66,14861],
#                [0.53,0.74,0.62,14861], [0.53,0.68,0.59,14861],
#                [0.64,0.62,0.63,14861]]
#micro_avg    = [[0.50,0.50,0.50,30000], [0.50,0.50,0.50,30000],
#                [0.55,0.55,0.55,30000], [0.54,0.54,0.54,30000],
#                [0.64,0.64,0.64,30000]]
#macro_avg    = [[0.25,0.50,0.33,30000], [0.65,0.50,0.50,30000],
#                [0.56,0.55,0.53,30000], [0.54,0.54,0.53,30000],
#                [0.64,0.64,0.64,30000]]
#weighted_avg = [[0.25,0.50,0.33,30000], [0.65,0.50,0.33,30000],
#                [0.56,0.55,0.53,30000], [0.54,0.54,0.53,30000],
#                [0.64,0.64,0.64,30000]]


c = [0.01, 0.1, 1, 10, 100]
## [class zero, class one, micro_avg, macro_avg, weighted_avg] for each C
precision = [[0.00,0.50,0.50,0.25,0.25], [0.80,0.50,0.50,0.65,0.65],
             [0.58,0.53,0.55,0.56,0.56], [0.56,0.53,0.54,0.54,0.54],
             [0.64,0.64,0.64,0.64,0.64]]
recall    = [[0.00,1.00,0.50,0.50,0.50], [0.00,1.00,0.50,0.50,0.50],
             [0.36,0.74,0.55,0.55,0.55], [0.39,0.68,0.54,0.54,0.54],
             [0.65,0.62,0.64,0.64,0.64]]
f1_score  = [[0.00,0.66,0.50,0.33,0.33], [0.01,0.66,0.50,0.50,0.33],
             [0.45,0.62,0.55,0.53,0.53], [0.46,0.59,0.54,0.53,0.53],
             [0.65,0.63,0.64,0.64,0.64]]
support   = [[15139,15139,15139,15139,15139], [14861,14861,14861,14861,14861],
             [30000,30000,30000,30000,30000], [30000,30000,30000,30000,30000],
             [30000,30000,30000,30000,30000]]

plt.figure(0)
f, axarr = plt.subplots(3, sharex=True)
axarr[0].semilogx(c, precision[0])
axarr[0].set(ylabel='Precision')
axarr[0].grid(True)
axarr[1].semilogx(c, recall[0], color='r')
axarr[1].set(ylabel='Recall')
axarr[1].grid(True)
axarr[2].semilogx(c, f1_score[0], color='k')
axarr[2].set(ylabel='F1 score', xlabel='C value')
axarr[2].grid(True)
f.suptitle('Class Zero Scores for SVM with Different C values')

plt.figure(1)
f, axarr = plt.subplots(3, sharex=True)
axarr[0].semilogx(c, precision[1])
axarr[0].set(ylabel='Precision')
axarr[0].grid(True)
axarr[1].semilogx(c, recall[1], color='r')
axarr[1].set(ylabel='Recall')
axarr[1].grid(True)
axarr[2].semilogx(c, f1_score[1], color='k')
axarr[2].set(ylabel='F1 score', xlabel='C value')
axarr[2].grid(True)
f.suptitle('Class One Scores for SVM with Different C values')

plt.figure(2)
f, axarr = plt.subplots(3, sharex=True)
axarr[0].semilogx(c, precision[2])
axarr[0].set(ylabel='Precision')
axarr[0].grid(True)
axarr[1].semilogx(c, recall[2], color='r')
axarr[1].set(ylabel='Recall')
axarr[1].grid(True)
axarr[2].semilogx(c, f1_score[2], color='k')
axarr[2].set(ylabel='F1 score', xlabel='C value')
axarr[2].grid(True)
f.suptitle('Micro Average Scores for SVM with Different C values')

plt.figure(3)
f, axarr = plt.subplots(3, sharex=True)
axarr[0].semilogx(c, precision[3])
axarr[0].set(ylabel='Precision')
axarr[0].grid(True)
axarr[1].semilogx(c, recall[3], color='r')
axarr[1].set(ylabel='Recall')
axarr[1].grid(True)
axarr[2].semilogx(c, f1_score[3], color='k')
axarr[2].set(ylabel='F1 score', xlabel='C value')
axarr[2].grid(True)
f.suptitle('Macro Average Scores for SVM with Different C values')

plt.figure(4)
f, axarr = plt.subplots(3, sharex=True)
axarr[0].semilogx(c, precision[4])
axarr[0].set(ylabel='Precision')
axarr[0].grid(True)
axarr[1].semilogx(c, recall[4], color='r')
axarr[1].set(ylabel='Recall')
axarr[1].grid(True)
axarr[2].semilogx(c, f1_score[4], color='k')
axarr[2].set(ylabel='F1 score', xlabel='C value')
axarr[2].grid(True)
f.suptitle('Weighted Average Scores for SVM with Different C values')

###############################################################################
confusion_matrix = [[[0,15139],[0,14861]], [[55,15084],[14,14847]], 
                    [[5508,9631],[3938,10923]], [[5968,9171],[4694,10167]],
                    [[9903,5236],[5619,9242]]]
for i in range(len(confusion_matrix)):
    plt.figure(i+6)
    plot_confusion_matrix(np.asarray(confusion_matrix[i]), classes=[0, 1],
                          title='Confusion Matrix for SVM with C = {0:g}'.format(c[i]))

plt.show()