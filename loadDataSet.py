import numpy as np
from preProcessDatabase import treatDatabase2

def loadDatas():
#INIT
    DatabaseTrain = []
    DatabaseTest = []
    pathDatabase = '/home/jean/Documentos/senai-sc/dados/'
    Labels = np.load(pathDatabase + 'Classes.npy', allow_pickle=True)
#DataSet Train
    DatabaseTrain.append(treatDatabase2(np.load(pathDatabase + 'Dados_1.npy', allow_pickle=True),Labels))
    DatabaseTrain.append(treatDatabase2(np.load(pathDatabase + 'Dados_2.npy', allow_pickle=True),Labels))
    DatabaseTrain.append(treatDatabase2(np.load(pathDatabase + 'Dados_3.npy', allow_pickle=True),Labels))
    DatabaseTrain.append(treatDatabase2(np.load(pathDatabase + 'Dados_4.npy', allow_pickle=True),Labels))    
    DatabaseTest.append(treatDatabase2(np.load(pathDatabase + 'Dados_5.npy', allow_pickle=True),Labels))
        
    result = {          
      'trainDataSet': DatabaseTrain, 
      'testDataSet': DatabaseTest  
    }     
    return result
