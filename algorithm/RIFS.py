'''
@File   :   RIFS.py
@Time   :   2021-1-2
@modify_time : 2021-1-15
@Author :   wang / gao
'''
from algorithm.utils import *

#sIFS called in algorithm RIFS
def sIFS(k, D, F, Case):
    # k is the start point
    # D means the decreasetime
    # F is the total data
    # Case is the label
    # fix bug for feature boundary

    SubF = list()
    SubF.append(F[k])

    a = Accuracy(SubF, Case)
    nxt = k + 1
    BestFS = SubF
    Dtimes = 0
    while Dtimes < D and nxt < len(F):
        if a == 1.0:
            break
        SubF.append(F[nxt])

        b = Accuracy(SubF, Case)

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


#execute RIFS algorithm in parallel
def sIFS_parallel(para):
    BestFS = sIFS(para[0], para[1], para[2],para[3])
    result = [Accuracy(BestFS, para[3]), BestFS, para[0]]
    return result


#RIFS
def RIFS(start, K, D, F, Case):
    # Modify code logic
    Solution = list()
    for i in range(0, K):
        start = random.randint(0, 0.45 * len(F))
        Solution.append(sIFS(start, D, F, Case))
    BestFS = Solution[0]
    for i in range(0, len(Solution)):
        if Accuracy(BestFS, Case) < Accuracy(Solution[i], Case):
            BestFS = Solution[i]
    result = [Accuracy(BestFS, Case), BestFS, start]
    return result
