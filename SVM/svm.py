import csv
import numpy as np
import scipy as sp
import pandas as pd
from topic import topic
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.svm import SVC
from gensim.models import ldamodel
import features
import pickle

comments = []
sarc_set = set()
features_list_train = []
features_list_test = []
data = pd.read_csv('./train-balanced-sarcasm.csv')
data.dropna(subset = ['comment'], inplace = True)

#topic model
#topic_model = topic(numTopics = 200, alpha = 'symmetric', model = "topics.model", dictionary = "dictionary.model")
topic_model = topic(numTopics = 200, alpha = 'symmetric')
topic_model.generate(data['comment'])

'''
infile = open("feature_vec_train.p",'rb')
feature_vec_train = pickle.load(infile)
infile.close()

infile = open("feature_vec_test.p",'rb')
feature_vec_test = pickle.load(infile)
infile.close()

infile = open("y_train.p",'rb')
feature_vec_test = pickle.load(infile)
infile.close()

infile = open("y_test.p",'rb')
feature_vec_test = pickle.load(infile)
infile.close()

'''

X_train, X_test, y_train, y_test = \
        train_test_split(data['comment'], data['label'], test_size = 0.3)

for comment in X_train[:50000]:
	features_list_train.append(features.get_features(comment, topic_model))

for comment in X_test[:30000]:
	features_list_test.append(features.get_features(comment, topic_model))

vec = DictVectorizer()

feature_vec_train = vec.fit_transform(np.array(features_list_train))
feature_vec_test = vec.transform(np.array(features_list_test))

file = open("feature_vec_train.p", 'wb')
pickle.dump(feature_vec_train, file)
file.close()

file = open("y_train.p", 'wb')
pickle.dump(y_train[:50000])
file.close()

file = open("feature_vec_test.p", 'wb')
pickle.dump(feature_vec_test, file)
file.close()

file = open("y_test.p", "wb")
pickle.dump(y_test[:30000])
file.close()



print "Training"
classifier = SVC()
classifier.fit(feature_vec_train, y_train[:50000])

file = open("classifier.p", 'wb')
pickle.dump(classifier, file)
file.close()

print "Testing"
y_pred = classifier.predict(feature_vec_test)

print confusion_matrix(y_test[:30000], y_pred)

print classification_report(y_test[:30000], y_pred)

