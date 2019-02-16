'''
@rana
multinomial naive bayes training code
'''

import csv
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import features_disc as features
import pickle
import time

start_time = time.time()

feature_file_train = "feature_vec_train_mnb.p"
feature_file_test = "feature_vec_test_mnb.p"
label_file_train = 'label_list_train_mnb.p'
label_file_test = 'label_list_test_mnb.p'
classifier_file = "mnb_classifier.p"

comments = []
train_features = {}
test_features = {}
features_list_train = []
features_list_test = []

data = pd.read_csv('./train-balanced-sarcasm.csv', names=['label', 'comment'], usecols=[0, 1], header=0)

print("Got comments")

_, X, _, Y = train_test_split(data['comment'], data['label'], test_size = 0.099)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

file = open(label_file_train, 'wb')
pickle.dump(Y_train, file)
file.close()

file = open(label_file_test, 'wb')
pickle.dump(Y_test, file)
file.close()

vec = DictVectorizer()

for comment in X_train:
        features.get_features(train_features, comment)
        features_list_train.append(train_features)
feature_vec_train = vec.fit_transform(np.array(features_list_train)).toarray()

file = open(feature_file_train, 'wb')
pickle.dump(feature_vec_train, file)
file.close()

for comment in X_test:
        features.get_features(test_features, comment)
        features_list_test.append(test_features)
feature_vec_test = vec.fit_transform(np.array(features_list_test)).toarray()

file = open(feature_file_test, 'wb')
pickle.dump(feature_vec_test, file)
file.close()

print("Got training features")

#feature_vec_train = pickle.load(open(feature_file_train, 'rb'))
#Y_train = pickle.load(open(label_file_train, 'rb'))

print ("Training")
classifier = MultinomialNB()
classifier.fit(feature_vec_train, Y_train)

file = open(classifier_file, 'wb')
pickle.dump(classifier, file)
file.close()

print("Training done")

elapsed_time = time.time() - start_time
print("Elapsed time: %d" %(elapsed_time))
