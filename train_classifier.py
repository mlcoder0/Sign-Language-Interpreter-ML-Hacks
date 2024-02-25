# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:29:47 2024

https://github.com/computervisioneng/sign-language-detector-python/blob/master/train_classifier.py

@author: Hoysala
"""

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#
data_dict = pickle.load(open('./data.pickle', 'rb'))

print(data_dict.keys())
#print(data_dict)

#RF classifier needs numpy data
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# separate out the train and test data. Suffle and stratify to increase the randomness in the training dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()