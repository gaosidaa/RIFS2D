'''
@File   :   process_data.py
@Time   :   2021-1-2 
@Author :   wang
'''
import global_var
import pandas as pd

from scipy import stats

def process_data_and_label():
    data_path = global_var.get_data_path()
    class_path = global_var.get_class_path()
    #file of data can be set here
    test = pd.read_csv(data_path)
    F = test.values.tolist()
    #file of class can be set here
    test = pd.read_csv(class_path)
    CC = test.values.tolist()
    Case = []
    for i in range(0, len(CC)):
        if CC[i][2] == 'P':
            Case.append(1)
        else:
            Case.append(0)

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
    F.sort(reverse=False)

    Feature = list()
    for f in F:
        Feature.append(f[1])
        del f[0:2]

    return Case,F,Feature