from scipy.stats import pearsonr

import numpy as np
import pandas as pd
import sklearn
import random
import csv

from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from multiprocessing import Pool

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

'''
#calculate
def Pear(F):
    p = list()
    l = len(F)
    if l == 1:
        return 1.0
    for i in range(0, l):
        for j in range(i + 1, l):
            x = pearsonr(F[i], F[j])
            # print(abs(x[0]))
            p.append(abs(x[0]))
    pAvg = np.mean(p)
    return pAvg
'''


#calculate average pearsonr between subblocks
def cc_between_block(F1, F2):
    l1 = len(F1)
    l2 = len(F2)
    c = list()
    for i in range(0, l1):
        for j in range(0, l2):
            pccs = pearsonr(F1[i], F2[j])
            c.append(abs(pccs[0]))
    cAvg = np.mean(c)
    return cAvg

#RIFS
def RIFS(start, K, D):
    Solution = list()
    for i in range(0, K):
        # random.seed(0)
        # k = random.randint(0, len(F))
        # k = int(len(F) * 0.45)
        # print('k =',k)
        Solution.append(sIFS(start, D))
    BestFS = Solution[0]
    for i in range(0, len(Solution)):
        if Accuracy(BestFS) < Accuracy(Solution[i]):
            BestFS = Solution[i]
    result = [Accuracy(BestFS), BestFS, start]
    return result

#RIFS called in function fcluster
def RIFS_fcluster(start, K, D):
    Solution = list()
    for i in range(0, K):
        # random.seed(0)
        # k = random.randint(0, len(F))
        # k = int(len(F) * 0.45)
        # print('k =',k)
        Solution.append(sIFS(start, D))
    BestFS = Solution[0]
    for i in range(0, len(Solution)):
        if Accuracy(BestFS) < Accuracy(Solution[i]):
            BestFS = Solution[i]
    return BestFS


#execute RIFS algorithm in parallel
def RIFS_parallel(para):
    return RIFS(para[0], para[1], para[2])

#sIFS called in algorithm RIFS
def sIFS(k, D):
    SubF = list()
    SubF.append(F[k])

    a = Accuracy(SubF)
    # print(type(a))

    nxt = k + 1
    BestFS = SubF
    Dtimes = 0
    while Dtimes < D and nxt < len(F):
        if a == 1.0:
            break
        SubF.append(F[nxt])

        b = Accuracy(SubF)

        if b <= a:
            Dtimes += 1
        else:
            a = b
            BestFS = SubF
            Dtimes = 0
        nxt += 1
    while Dtimes > 0:
        BestFS.pop()
        Dtimes -= 1
    return BestFS

#calculate Accuracy of sets of features
def Accuracy(F):
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
def Accuracy_all_classifier(F):
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
def Accuracy_one_time(F):
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

#delete some of chosen features to get a better performance
def DEL(Sub,top):
    sub = Sub
    numFeature = len(Sub)
    # print(numFeature)

    while (1):
        if (numFeature < 2):
            break
        # print(Accuracy(sub))

        standardAcc = Accuracy(sub)
        subAcc = list()
        for i in range(0, len(sub)):
            f = sub[0]
            del sub[0]
            subAcc.append([Accuracy_one_time(sub), i])
            sub.append(f)
        subAcc.sort(reverse=True)
        if standardAcc >= subAcc[0][0]:
            break
        else:
            p = subAcc[0][1]
            del sub[p]
            del top[p]
    return [Accuracy_all_classifier(sub),len(sub),top]

'''
def Merge_To_One(sub1, sub2):
    sub = list()
    for i in range(0, len(sub1)):
        sub.append(sub1[i])
    for i in range(0, len(sub2)):
        sub.append(sub2[i])
    return sub
'''

#calculate pearsonr between subblocks after filtering
def cc_between_block_after_filter(Sub1, Sub2):
    # print("ok")
    sub1 = list()
    sub2 = list()
    Sub = list()

    for i in range(0, len(Sub1)):
        sub1.append(Sub1[i])
        Sub.append(Sub1[i])
    for i in range(0, len(Sub2)):
        sub2.append(Sub2[i])
        Sub.append(Sub2[i])
    sub = Sub
    numFeature = len(Sub)
    # print(numFeature)

    while (1):
        if (numFeature < 2):
            break
        # print(Accuracy(sub))

        standardAcc = Accuracy_one_time(sub)
        subAcc = list()
        for i in range(0, len(sub)):
            f = sub[0]
            del sub[0]
            subAcc.append([Accuracy_one_time(sub), i])
            sub.append(f)
        subAcc.sort(reverse=True)
        if standardAcc >= subAcc[0][0]:
            break
        else:
            p = subAcc[0][1]
            f = sub[p]
            del sub[p]
            whether = 0
            for i in range(0, len(sub1)):
                if (f == sub1[i]):
                    whether = 1
                    del sub1[i]
                    # print("ok")
                    break
            if (whether == 0):
                for i in range(0, len(sub2)):
                    if (f == sub2[i]):
                        del sub2[i]
                        # print("ok")
                        break
    # the pearsonr Number after filter
    return cc_between_block(sub1, sub2)

#calculate pearsonr between subblocks in parallel
def cc_between_block_after_filter_parallel(para):
    result = list()
    result.append(para[0])
    result.append(para[1])
    result.append(cc_between_block_after_filter(para[2], para[3]))
    return result

#cluster subblocks of features
def fcluster(start, maxNumber, subGraph, topStart,decreaseTime):
    startPoint = start
    relTemp = []
    countn = 0
    relTemp.append(startPoint)
    presentSet = set(subGraph[startPoint])
    while len(presentSet) != 0 and countn <= maxNumber:
        presentList = list(presentSet)
        presentList.sort(reverse=False)
        nxt = presentList[0]
        presentSet = presentSet.intersection(set(subGraph[nxt]))
        relTemp.append(nxt)
        countn = countn + 1
    sub = list()
    top = list()
    for j in range(len(relTemp)):
        tmpSub = RIFS_fcluster(topStart[relTemp[j]], 1, decreaseTime)
        totalFeatures = len(tmpSub)
        for i in range(0,totalFeatures):
            addNum = topStart[relTemp[j]]+i
            if addNum not in top:
                top.append(addNum)
                sub.append(tmpSub[i])
    rel = DEL(sub,top)
    return [start, rel[0],rel[1],rel[2]]


#cluster subblocks in parallel
def fcluster_parallel(para):
    return fcluster(para[0], para[1], para[2], para[3],para[4])

data_path = "./data/TCGA_BRCA_FPKM_select.csv"
class_path = "./clinical/TCGA_BRCA_FPKM_class.csv"
# test = pd.read_csv('~/Documents/Research/RIFS/dataset/data/ALL3.csv')
#file of data can be set here
test = pd.read_csv(data_path)
F = test.as_matrix().tolist()
# test = pd.read_csv('~/Documents/Research/RIFS/dataset/class/ALL3class.csv')
#file of class can be set here
test = pd.read_csv(class_path)
CC = test.as_matrix().tolist()
Case = []
# print(F[0])
for i in range(0, len(CC)):
    if CC[i][2] == 'P':
        Case.append(1)
    else:
        Case.append(0)
# print('Case =',Case)

#calculate p-value in the t-test of each feature
for f in F:
    F1 = list()
    F2 = list()
    for i in range(0, len(Case)):
        if Case[i] == 1:
            F1.append(f[i + 1])
        else:
            F2.append(f[i + 1])
    p = stats.ttest_ind(F1, F2, equal_var=False).pvalue
    f.insert(0, p)

# print(F[0])
#reverse = False (in increasing order)
F.sort(reverse=False)
# print(F[0])

Feature = list()
for f in F:
    Feature.append(f[1])
    del f[0:2]
# print('F[0] =',F[0],len(F[0]),len(Case))


acc_save_path = "./acc/BRCA_acc_12_28.csv"
distance_save_path = "./distance/BRCA_12_28.csv"
result_save_path = "./result/BRCA_12_28.csv"



if __name__ == "__main__":
    #number of CPU which run in parallel can be set here
    cpuNumber = 3
    totalNumber=len(F)
    topNumber = int(0.01*totalNumber)
    '''
    totalNumber = 8
    '''
    pool = Pool(cpuNumber)
    para = list()
    #parameter decreaseTime can be set here
    D=5
    for i in range(0, totalNumber):
        paraTemp = list()
        #sIFS下降的次数，第一个参数
        Times = 1
        paraTemp.append(i)
        paraTemp.append(Times)
        paraTemp.append(D)
        para.append(paraTemp)
    result = pool.map(RIFS_parallel, para)
    result.sort(reverse=True)
    '''
    topNumber = int(0.01*totalNumber)
    topNumber = totalNumber
    '''
    topStart = list()
    topFeature = list()
    for i in range(0, topNumber):
        topStart.append(result[i][2])
        topFeature.append(result[i][1])
    # write the result of Accuracy of each subblock to the file below
    out = open(acc_save_path, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(["start"])
    for ts in topStart:
        csv_write.writerow([ts])
    out.close()
    para = list()
    for i in range(0, topNumber):
        for j in range(i+1, topNumber):
            para_temp = list()
            para_temp.append(i)
            para_temp.append(j)
            para_temp.append(topFeature[i])
            para_temp.append(topFeature[j])
            para.append(para_temp)
    pool = Pool(cpuNumber)
    result = pool.map(cc_between_block_after_filter_parallel, para)
    pool.close()
    pool.join()
    #write the result of pearsonr between chosen subblocks can be set here
    out = open(distance_save_path, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(["start1", "start2", "pearsonr"])
    for r in result:
        r[2] = min(1, r[2])
        csv_write.writerow(r)
    out.close()
    csv_pearsonr = pd.read_csv(distance_save_path)
    csv_start = pd.read_csv(acc_save_path)
    # record the start number
    topStart = list()
    for i in range(0, len(csv_start)):
        topStart.append(csv_start.loc[i][0])
    # calculate the distance average/median
    # record the pearsonr
    pearsonrNumber = list()
    pearsonrMatrix = [[0 for col in range(topNumber)] for row in range(topNumber)]
    for i in range(0, len(csv_pearsonr)):
        if csv_pearsonr.loc[i][2] != 1:
            pearsonrNumber.append(csv_pearsonr.loc[i][2])
        pearsonrMatrix[int(csv_pearsonr.loc[i][0])][int(csv_pearsonr.loc[i][1])] = csv_pearsonr.loc[i][2]
        pearsonrMatrix[int(csv_pearsonr.loc[i][1])][int(csv_pearsonr.loc[i][0])] = csv_pearsonr.loc[i][2]
    #pearsonrMedian = np.median(pearsonrNumber)
    pearsonrAverage=np.mean(pearsonrNumber)
    # create the graph
    distanceMatrix = [[1 for col in range(0, topNumber)] for row in range(0, topNumber)]
    # delete the edge
    for i in range(0, topNumber):
        for j in range(0, topNumber):
            #parameter percentMedian can be set here
            if pearsonrMatrix[i][j] > pearsonrAverage * 0.35:
                distanceMatrix[i][j] = 0
                distanceMatrix[j][i] = 0
    # find the sub graph
    subGraph = list()
    for i in range(0, topNumber):
        temp = list()
        for j in range(0, topNumber):
            if distanceMatrix[i][j] == 1:
                temp.append(j)
        subGraph.append(temp)
    total = topNumber
    # total=1
    # the maximum number of combinations
    #parameter maxNumber can be set here
    maxNumber = 5
    para = list()
    for i in range(0, total):
        tempPara = [i, maxNumber, subGraph, topStart,D]
        para.append(tempPara)
    if len(para) != 0:
        pool = Pool(cpuNumber)
        result = pool.map(fcluster_parallel, para)
        pool.close()
        pool.join()
        #write the final result the file below
        out = open(result_save_path, 'a', newline="")
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(["start","dtree","KNN","lsvc","lr","gnb","Number","subblock"])
        for i in range(len(result)):
            temp = [result[i][0]]
            for j in range(0,5):
                temp.append(result[i][1][j])
            temp.append(result[i][2])
            temp.append(result[i][3])
            csv_write.writerow(temp)
        out.close()


















