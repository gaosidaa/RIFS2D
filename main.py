'''
@File   :   main.py
@Time   :   2021-1-3
@modify_time : 2021-1-15
@modify_time : 2021-1-22
@Author :   wang/gao
'''

import os
import argparse
import global_var
import pandas as pd
import numpy as np

from input.process_data import process_data_and_label
from algorithm.RIFS import sIFS_parallel
from algorithm.RIFS2D import cc_between_block_after_filter_parallel, fcluster_parallel
from output.output_file import output_acc,output_distance,output_result,output_report
from multiprocessing import Pool
from configparser import ConfigParser

import warnings
import sklearn

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

if __name__ == "__main__":
    #set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=str,default="./data/ALL1.csv", help="path of data")
    parser.add_argument("--cp", type=str,default="./class/ALL1class.csv", help="path of class")
    parser.add_argument("--config", type=str, default="./config/config.cfg", help="path of config")

    args = parser.parse_args()

    #set global variable
    global_var.set_data_path(args.dp)
    global_var.set_class_path(args.cp)
    Case, F, Feature = process_data_and_label()

    #read config file
    config_path = args.config
    config_parser = ConfigParser()
    config_parser.read(config_path)

    #set path
    output_path = str(config_parser.get("write_path", "rel_save_dir"))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_name = "ALL1_1_4.csv"
    name = args.dp
    file_name = name[7:-4]
    #print(file_name)

    acc_name = file_name + "_acc.csv"
    acc_save_dir = os.path.join(output_path, "acc/")
    if not os.path.exists(acc_save_dir):
        os.makedirs(acc_save_dir)
    acc_save_path = os.path.join(acc_save_dir, acc_name)

    distance_name = file_name + "_distance.csv"
    distance_save_dir = os.path.join(output_path,"distance/")
    if not os.path.exists(distance_save_dir):
        os.makedirs(distance_save_dir)
    distance_save_path = os.path.join(distance_save_dir, distance_name)

    rel_name = file_name + "_rel.csv"
    result_save_dir = os.path.join(output_path,"rel/")
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    result_save_path = os.path.join(result_save_dir, distance_name)
    report_save_path = os.path.join(output_path, "report.txt")

    cpuNumber = int(config_parser.get("parameters", "cpu_number"))
    '''
    totalNumber = 5
    topNumber = totalNumber
    '''
    totalNumber = len(F)
    totalPercent = float(config_parser.get("default", "total_percent"))
    topNumber = int(totalPercent* totalNumber)





    #calcute
    pool = Pool(cpuNumber)
    para = list()
    D = int(config_parser.get("parameters","decrease_times"))
    for i in range(0, totalNumber):
        paraTemp = list()
        Times = 1
        paraTemp.append(i)
        #paraTemp.append(Times)
        paraTemp.append(D)
        paraTemp.append(F)
        paraTemp.append(Case)
        para.append(paraTemp)
    result = pool.map(sIFS_parallel, para)
    result.sort(reverse=True)

    topStart = list()
    topFeature = list()
    for i in range(0, topNumber):
        topStart.append(result[i][2])
        topFeature.append(result[i][1])

    output_acc(acc_save_path, topStart)

    para = list()
    for i in range(0, topNumber):
        for j in range(i + 1, topNumber):
            para_temp = list()
            para_temp.append(i)
            para_temp.append(j)
            para_temp.append(topFeature[i])
            para_temp.append(topFeature[j])
            para_temp.append(Case)
            para.append(para_temp)
    pool = Pool(cpuNumber)
    result = pool.map(cc_between_block_after_filter_parallel, para)
    pool.close()
    pool.join()

    output_distance(distance_save_path,result)

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
    pearsonrAverage = np.mean(pearsonrNumber)
    # create the graph
    distanceMatrix = [[1 for col in range(0, topNumber)] for row in range(0, topNumber)]
    # delete the edge
    percentMedian = float(config_parser.get("parameters","percent_median"))
    for i in range(0, topNumber):
        for j in range(0, topNumber):
            # parameter percentMedian can be set here
            if pearsonrMatrix[i][j] > pearsonrAverage * percentMedian:
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
    # the maximum number of combinations
    # parameter maxNumber can be set here
    maxNumber = int(config_parser.get("parameters","max_number"))
    para = list()
    for i in range(0, total):
        tempPara = [i, maxNumber, subGraph, topStart, D, F, Case]
        para.append(tempPara)
    if len(para) != 0:
        pool = Pool(cpuNumber)
        result = pool.map(fcluster_parallel, para)
        pool.close()
        pool.join()

        output_result(result_save_path,result)
    output_report(report_save_path)