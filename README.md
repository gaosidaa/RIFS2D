# RIFS2D
RIFS2D is an improved feature selection algorithm based on RIFS used in binary classification problems about disease diagnosis.

**necessary packages**

* numpy
* pandas
* sklearn
* scipy

**environment**
* Python 3.7
* conda 4.8

**folders**
* algorithm, input, output: source code
* config: config file
* data: dataset
* class: class label for samples
* result: output result

**examples**
* dataset: data/ALL1.csv, class/ALL1class.csv
* config: config/config.cfg

**how to run**

`python main.py --dp ./data/ALL1.csv --cp ./class/ALL1class.csv --config ./config/config.cfg`

or 

`python3 main.py --dp ./data/ALL1.csv --cp ./class/ALL1class.csv --config ./config/config.cfg`


The result will be saved in directory shown in config.cfg.
Details about result is shown in report.txt of result directory.  

Contact wangpuli@hotmail.com or gaosidaa@gmail.com for more questions.
