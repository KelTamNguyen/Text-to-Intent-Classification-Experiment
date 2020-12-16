# -*- coding: utf-8 -*-
"""
CSC2730 Final Project
Written by Kelvin Nguyen, William Daniels
"""

from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Read the dataset using pandas
X = pd.read_csv('sample.csv')

# Assign future phrases to 'patterns' and future tags to 'intents'
patterns = list(X['phrases'])
intents = list(X['tags'])

# Zip them together for shuffling
fullList = list(zip(patterns, intents))

# Shuffle the entries so when we split the data it is equally representative
shuffle(fullList)

# Put back into separate lists
phrases, tags = zip(*fullList)
phrases, tags = list(phrases), list(tags)

# Add the phrases of the dataset to one list and the corresponding tags to another list
for listing in fullList:
    for n, string in enumerate(listing):
        if n % 2 == 0:
            phrases.append(string)
        else:
            tags.append(string)

# Split our data into 80% training and 20% testing data
train_X, test_X, train_y, test_y = train_test_split(phrases, tags, test_size=.2)

print("First model is a pipeline that includes a Tf-idf Vectorizer with a SVC model")

# Create a TfidfVectorizer and a SVC to create a bag of words representation for the data
vect = TfidfVectorizer()
svc = SVC(kernel='poly', random_state=22)

# Create a pipeline that 'vect' and 'svc' feed into
pipeModel = Pipeline([
    ('vect', vect),
    ('svc', svc)
])

# Fit the model on the training data
pipeModel.fit(train_X, train_y)

# Predict on the testing phrases
p1 = pipeModel.predict(test_X)

# Print model accuracy
print("Accuracy for our pipeline model: ", accuracy_score(test_y, p1))

print("\nLet's try different hyperparameters for our Pipeline model")

for c in [3, 5, 10]:
    for k in ['linear', 'poly', 'rbf']:
        vect1 = TfidfVectorizer()
        svc1 = SVC(kernel=k, C=c, random_state=22)

        pipeModel1 = Pipeline([
            ('vect', vect1),
            ('svc', svc1)
        ])

        pipeModel1.fit(train_X, train_y)
        p1 = pipeModel1.predict(test_X)

        print("Accuracy where C={}, k={}: {}".format(c, k, accuracy_score(test_y, p1)))
print()


for minDF in [0.005, 0.01]:
    for maxDF in [.995, .9995]:
        vect2 = TfidfVectorizer(min_df=minDF, max_df=maxDF)
        svc2 = SVC(kernel='rbf', C=10, random_state=22)

        pipeModel2 = Pipeline([
            ('vect', vect2),
            ('svc', svc2)
        ])

        pipeModel2.fit(train_X, train_y)
        p2 = pipeModel2.predict(test_X)

        print("Accuracy where min_df={} and max_df={}: {}".format(minDF, maxDF, accuracy_score(test_y, p2)))
print()


print("Next model is a K-Nearest Neighbors model")

# Transform the data into vectors
train_X = vect.fit_transform(train_X)
test_X = vect.transform(test_X)

print("\nLet's try different hyperparameters for our K-Nearest Neighbors model")


for neighbors in [3, 6, 9]:
    for weights in ['distance', 'uniform']:
        knnModel = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)
        knnModel.fit(train_X, train_y)
        p3 = knnModel.predict(test_X)
        print("Accuracy for knn model with {} neighbors and a {} weight function:{}".format(neighbors, weights,
                                                                                            accuracy_score(test_y, p3)))
