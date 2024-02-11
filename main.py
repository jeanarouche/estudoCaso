
from mainHMM import init_hmm

from loadDataSet import loadDatas
####################################
dataSet = loadDatas()
train =  dataSet['trainDataSet']
test = dataSet['testDataSet']


orderHmm = 2
hmm = init_hmm(train,test,orderHmm)


