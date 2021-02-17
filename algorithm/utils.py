'''
@File   :   utils.py
@Time   :   2021-1-2 
@modify_time : 2021-1-15
@Author :   wang / gao
'''
import numpy as np
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#calculate Accuracy of sets of features
def Accuracy(F, Case):
    y = Case
    _cv = 10
    macc = 0
    Cross_valid_data = np.array(F).T.tolist()
    Positive = list()
    Negative = list()
    # divide the positive and negative features
    positive_count = 0
    negative_count = 0
    if (isinstance(Cross_valid_data, float)): return 0
    if (len(Cross_valid_data) == 0): return 0
    for i in range(0, len(y)):
        if (y[i] == 1):
            Positive.append(Cross_valid_data[i])
            positive_count = positive_count + 1
        else:
            Negative.append(Cross_valid_data[i])
            negative_count = negative_count + 1
    result = [0] * 5
    for i in range(0, 20):
        rand_positive = list(range(0, positive_count))
        random.shuffle(rand_positive)
        rand_negative = list(range(0, negative_count))
        random.shuffle(rand_negative)
        Acc = [0] * 5
        for j in range(1, 11):
            x_train = list()
            y_train = list()
            x_test = list()
            y_test = list()
            for k in range(0, positive_count):
                if (k >= (int)((j - 1) * 0.1 * positive_count) and k <= (int)(j * 0.1 * positive_count) - 1):
                    x_test.append(Positive[rand_positive[k]])
                    y_test.append(1)
                else:
                    x_train.append(Positive[rand_positive[k]])
                    y_train.append(1)
            for k in range(0, negative_count):
                if (k >= (int)((j - 1) * 0.1 * negative_count) and k <= (int)(j * 0.1 * negative_count) - 1):
                    x_test.append(Negative[rand_negative[k]])
                    y_test.append(0)
                else:
                    x_train.append(Negative[rand_negative[k]])
                    y_train.append(0)
            Rel = list()
            dtree = DecisionTreeClassifier(max_depth=10)
            Rel.append(dtree)
            knn = KNeighborsClassifier(n_neighbors=3)
            Rel.append(knn)
            lsvc = LinearSVC()
            Rel.append(lsvc)
            lr = LogisticRegression(solver='liblinear', max_iter=5000)
            Rel.append(lr)
            gnb = GaussianNB()
            Rel.append(gnb)
            for k in range(0, 5):
                Rel[k].fit(x_train, y_train)
                y_predict = Rel[k].predict(x_test)
                correct = 0
                for m in range(0, len(y_predict)):
                    if (y_predict[m] == y_test[m]):
                        correct = correct + 1
                temp_acc = correct / len(y_predict)
                Acc[k] = Acc[k] + temp_acc
        for j in range(0, 5):
            result[j] = result[j] + (Acc[j] / _cv)
    for i in range(0, 5):
        result[i] = result[i] / 20
    return max(result)

#calculate Accuracy achieved by all five chosen classifiers
def Accuracy_all_classifier(F, Case):
    y = Case
    _cv = 10
    macc = 0
    Cross_valid_data = np.array(F).T.tolist()
    Positive = list()
    Negative = list()
    # divide the positive and negative features
    positive_count = 0
    negative_count = 0
    if (isinstance(Cross_valid_data, float)): return 0
    if (len(Cross_valid_data) == 0): return 0
    for i in range(0, len(y)):
        if (y[i] == 1):
            Positive.append(Cross_valid_data[i])
            positive_count = positive_count + 1
        else:
            Negative.append(Cross_valid_data[i])
            negative_count = negative_count + 1
    result = [0] * 5
    for i in range(0, 20):
        rand_positive = list(range(0, positive_count))
        random.shuffle(rand_positive)
        rand_negative = list(range(0, negative_count))
        random.shuffle(rand_negative)
        Acc = [0] * 5
        for j in range(1, 11):
            x_train = list()
            y_train = list()
            x_test = list()
            y_test = list()
            for k in range(0, positive_count):
                if (k >= (int)((j - 1) * 0.1 * positive_count) and k <= (int)(j * 0.1 * positive_count) - 1):
                    x_test.append(Positive[rand_positive[k]])
                    y_test.append(1)
                else:
                    x_train.append(Positive[rand_positive[k]])
                    y_train.append(1)
            for k in range(0, negative_count):
                if (k >= (int)((j - 1) * 0.1 * negative_count) and k <= (int)(j * 0.1 * negative_count) - 1):
                    x_test.append(Negative[rand_negative[k]])
                    y_test.append(0)
                else:
                    x_train.append(Negative[rand_negative[k]])
                    y_train.append(0)
            Rel = list()
            dtree = DecisionTreeClassifier(max_depth=10)
            Rel.append(dtree)
            knn = KNeighborsClassifier(n_neighbors=3)
            Rel.append(knn)
            lsvc = LinearSVC()
            Rel.append(lsvc)
            lr = LogisticRegression(solver='liblinear', max_iter=5000)
            Rel.append(lr)
            gnb = GaussianNB()
            Rel.append(gnb)
            for k in range(0, 5):
                Rel[k].fit(x_train, y_train)
                y_predict = Rel[k].predict(x_test)
                correct = 0
                for m in range(0, len(y_predict)):
                    if (y_predict[m] == y_test[m]):
                        correct = correct + 1
                temp_acc = correct / len(y_predict)
                Acc[k] = Acc[k] + temp_acc
        for j in range(0, 5):
            result[j] = result[j] + (Acc[j] / _cv)
    for i in range(0, 5):
        result[i] = result[i] / 20
    return result

#Accuracy only run three times
def Accuracy_one_time(F, Case):
    y = Case
    _cv = 10
    macc = 0
    Cross_valid_data = np.array(F).T.tolist()
    Positive = list()
    Negative = list()
    # divide the positive and negative features
    positive_count = 0
    negative_count = 0
    if (isinstance(Cross_valid_data, float)): return 0
    if (len(Cross_valid_data) == 0): return 0
    for i in range(0, len(y)):
        if (y[i] == 1):
            Positive.append(Cross_valid_data[i])
            positive_count = positive_count + 1
        else:
            Negative.append(Cross_valid_data[i])
            negative_count = negative_count + 1
    result = [0] * 5
    for i in range(0, 3):
        rand_positive = list(range(0, positive_count))
        random.shuffle(rand_positive)
        rand_negative = list(range(0, negative_count))
        random.shuffle(rand_negative)
        Acc = [0] * 5
        for j in range(1, 11):
            x_train = list()
            y_train = list()
            x_test = list()
            y_test = list()
            for k in range(0, positive_count):
                if (k >= (int)((j - 1) * 0.1 * positive_count) and k <= (int)(j * 0.1 * positive_count) - 1):
                    x_test.append(Positive[rand_positive[k]])
                    y_test.append(1)
                else:
                    x_train.append(Positive[rand_positive[k]])
                    y_train.append(1)
            for k in range(0, negative_count):
                if (k >= (int)((j - 1) * 0.1 * negative_count) and k <= (int)(j * 0.1 * negative_count) - 1):
                    x_test.append(Negative[rand_negative[k]])
                    y_test.append(0)
                else:
                    x_train.append(Negative[rand_negative[k]])
                    y_train.append(0)
            Rel = list()
            dtree = DecisionTreeClassifier(max_depth=10)
            Rel.append(dtree)
            knn = KNeighborsClassifier(n_neighbors=3)
            Rel.append(knn)
            lsvc = LinearSVC()
            Rel.append(lsvc)
            lr = LogisticRegression(solver='liblinear', max_iter=5000)
            Rel.append(lr)
            gnb = GaussianNB()
            Rel.append(gnb)
            for k in range(0, 5):
                Rel[k].fit(x_train, y_train)
                y_predict = Rel[k].predict(x_test)
                correct = 0
                for m in range(0, len(y_predict)):
                    if (y_predict[m] == y_test[m]):
                        correct = correct + 1
                temp_acc = correct / len(y_predict)
                Acc[k] = Acc[k] + temp_acc
        for j in range(0, 5):
            result[j] = result[j] + (Acc[j] / _cv)
    for i in range(0, 5):
        result[i] = result[i] / 3
    return max(result)