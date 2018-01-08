# -*- coding:utf-8 -*-

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from skmultilearn.ext import Meka

from skmultilearn.adapt import MLkNN

X, y = make_multilabel_classification(sparse=True, return_indicator='sparse')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

meka = Meka(
    meka_classifier="meka.classifiers.multilabel.LC",
    meka_classpath="F:\\mekalib\\",
    java_command="d:\\Program Files\\Java/jdk1.8.0_60\\jre\\bin\\java")

meka.fit(X_train, y_train)

predictions = meka.predict(X_test)

hamming_loss(y_test, predictions)