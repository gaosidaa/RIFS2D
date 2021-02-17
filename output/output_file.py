'''
@File   :   output_file.py
@Time   :   2021-1-3
@modify_time : 2021-1-20
@Author :   wang / gao
'''
import csv

def output_acc(acc_save_path, topStart):
    with open(acc_save_path, 'w') as out:
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(["start"])
        for ts in topStart:
            csv_write.writerow([ts])

def output_distance(distance_save_path, result):
    # write the result of pearsonr between chosen subblocks can be set here
    with open(distance_save_path, 'w', newline='') as out:
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(["start1", "start2", "pearsonr"])
        for r in result:
            r[2] = min(1, r[2])
            csv_write.writerow(r)

def output_result(result_save_path, result):
    # write the final result the file below
    with open(result_save_path, 'w', newline="") as out:
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(["start", "dtree", "KNN", "lsvc", "lr", "gnb", "Number", "subblock"])
        for i in range(len(result)):
            temp = [result[i][0]]
            for j in range(0, 5):
                temp.append(result[i][1][j])
            temp.append(result[i][2])
            temp.append(result[i][3])
            csv_write.writerow(temp)

def output_report(report_save_path):
    fp = open(report_save_path,'w')
    write_list = ["./result/acc/","start: rank in t-test and sort in unincreased order of Accuracy",
                  "./result/distance/","start1: t-test rank of one subblock","start2: t-test rank of the other subblock",
                  "pearsonr: PCC value between two subblocks",
                  "./result/rel/", "start: rank in t-test of starting subblock",
                  "dtree/KNN/lsvc/lr/gnb: Accuracy of five classifiers",
                  "Number: number of features in final solution",
                  "subblock: detailed features (t-test rank) of final solution"]
    for w in write_list:
        fp.write("{}\n".format(w))
