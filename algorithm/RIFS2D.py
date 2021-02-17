'''
@File   :   RIFS2D.py
@Time   :   2021-1-2
@modify_time : 2021-1-15
@Author :   wang / gao
'''
from algorithm.utils import *
from algorithm.RIFS import sIFS
from scipy.stats import pearsonr


#delete some of chosen features to get a better performance
def DEL(Sub,top,Case):
    sub = Sub
    numFeature = len(Sub)
    # print(numFeature)

    while (1):
        if (numFeature < 2):
            break

        standardAcc = Accuracy(sub, Case)
        subAcc = list()
        for i in range(0, len(sub)):
            f = sub[0]
            del sub[0]
            subAcc.append([Accuracy_one_time(sub, Case), i])
            sub.append(f)
        subAcc.sort(reverse=True)
        if standardAcc >= subAcc[0][0]:
            break
        else:
            p = subAcc[0][1]
            del sub[p]
            del top[p]
    return [Accuracy_all_classifier(sub, Case),len(sub),top]

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

#calculate pearsonr between subblocks after filtering
def cc_between_block_after_filter(Sub1, Sub2, Case):
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


    while (1):
        if (numFeature < 2):
            break

        standardAcc = Accuracy_one_time(sub, Case)
        subAcc = list()
        for i in range(0, len(sub)):
            f = sub[0]
            del sub[0]
            subAcc.append([Accuracy_one_time(sub, Case), i])
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
                    break
            if (whether == 0):
                for i in range(0, len(sub2)):
                    if (f == sub2[i]):
                        del sub2[i]
                        break

    return cc_between_block(sub1, sub2)

#calculate pearsonr between subblocks in parallel
def cc_between_block_after_filter_parallel(para):
    result = list()
    result.append(para[0])
    result.append(para[1])
    result.append(cc_between_block_after_filter(para[2], para[3],para[4]))
    return result

#cluster subblocks of features
def fcluster(start, maxNumber, subGraph, topStart,decreaseTime, F, Case):
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
        tmpSub = sIFS(topStart[relTemp[j]], decreaseTime, F, Case)
        totalFeatures = len(tmpSub)
        for i in range(0,totalFeatures):
            addNum = topStart[relTemp[j]]+i
            if addNum not in top:
                top.append(addNum)
                sub.append(tmpSub[i])
    rel = DEL(sub,top, Case)
    return [start, rel[0],rel[1],rel[2]]


#cluster subblocks in parallel
def fcluster_parallel(para):
    return fcluster(para[0], para[1], para[2], para[3],para[4],para[5],para[6])

