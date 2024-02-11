import numpy as np
from hmm import estVeross
from hmm import estLogBeta
from hmm import estLogAlpha
from hmm import estLogAlphaBeta
from hmm import estBaumWelch
from hmm import avalBaumWelch
from hmm import avalLogAlphaBeta
from hmm import avalVeross
from hmm import avalLogBeta
import json
from preProcessDatabase import baseClass
from preProcessDatabase import treatHistoryTrain
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings 
#Array da matriz de confusão
#confusion_matrix = metrics.confusion_matrix(actual, predicted)
#resultado acuracia recall f1-score
#print(classification_report(a, p))

def showHMM(result): 
    
    print("\n#############################################################")   
    print("Resultado da classe:", result['nClass'], " na base:", result['nBase'], " Tam:", result['lClass'])
    print("###############################################################")
    print("Tempo de processamento : ", round(result['tmp'],2) , " segundos" )    
    print("##################################")   
    print("Média (mu) : ", result['est_mu'][0], " ", result['est_mu'][1])
    print("###################################")   
    print("Variância (sigma2) :", result['est_sigma2'][0], " ", result['est_sigma2'][1])
    print("##########################################")   
    print("Probab inicial (delta):\n", result['est_delta'][0], " ", result['est_delta'][1])
    print("##########################################")   
    print("Probab transição (gamma):\n", result['est_gamma'][0][0], " ",result['est_gamma'][0][1], "\n", 
                                                result['est_gamma'][1][0], " ",result['est_gamma'][1][1] )            
    print("###############################################") 
    print("Max_Verossimilhança (mllk):", result['mllk']) 
    print("################################################") 
    print("AIC:", result['AIC']) 
    print("################################################") 
    print("BIC:", result['BIC']) 
    print("################################################") 
    print("Nº de iterações necessárias:", result['itr']) 
    print("################################################") 

def showResults(labelTest,predClassLabel):    
    warnings.filterwarnings('ignore')
    print("\n################ Confusion Matrix ####################\n")
    print(classification_report(np.array(labelTest), np.array(predClassLabel)  ))
    print("######################################################")
    
def analiseBetterTrain(accuracy,estTrain,labelTest,predClassLabel):        
    
    accuracyActual = accuracy
    labelTest = labelTest
    predClassLabel =  predClassLabel    
    data = estTrain                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                      
    try:        
        with open("history/accuracyLast.json", "r") as f:                  
          accuracyLast = json.load(f)     
          if(accuracyActual > accuracyLast):
              accuracyLast = accuracyActual                       
              file = open("history/accuracyLast.json", 'w')
              file.write(str(accuracyLast))   
              file.close()
              
              est_train = treatHistoryTrain(str(data))
              file = open("history/est_train.json", 'w')
              file.write(est_train)    
              file.close()
              
              labelTest = str(labelTest)
              file = open("history/labelTest.json", 'w')
              file.write(labelTest)    
              file.close() 
              
              predClassLabel = str(predClassLabel)
              file = open("history/predClassLabel.json", 'w')
              file.write(predClassLabel)    
              file.close()
              result = {          
               'estClassTrain': est_train, 
               'predClassLabel': predClassLabel,
               'labelTest': labelTest        
              } 
              return result
          else:
              with open("history/accuracyLast.json", "r") as f:                  
                      accuracyLast = json.load(f)
              with open("history/est_train.json", "r") as f:                  
                      est_train = json.load(f) 
              with open("history/labelTest.json", "r") as f:                  
                      labelTest = json.load(f)
              with open("history/predClassLabel.json", "r") as f:                  
                      predClassLabel = json.load(f)
              """print("Tudo", accuracyLast, "\n---", est_train, 
                          "\n---", labelTest,
                          "\n---", predClassLabel)"""    
              result = {          
                 'estClassTrain': est_train, 
                 'predClassLabel': predClassLabel,
                 'labelTest': labelTest        
              } 
              return result

    except (FileNotFoundError):    
        accuracyLast = 0
        if(accuracyActual > 0.0):
            accuracyLast = accuracyActual                
        file = open("history/accuracyLast.json", 'w')
        file.write(str(accuracyLast))   
        file.close()
        
        est_train = treatHistoryTrain(str(data))
        file = open("history/est_train.json", 'w')
        file.write(est_train)    
        file.close()
        
        labelTest = str(labelTest)
        file = open("history/labelTest.json", 'w')
        file.write(labelTest)    
        file.close() 
        
        predClassLabel = str(predClassLabel)
        file = open("history/predClassLabel.json", 'w')
        file.write(predClassLabel)    
        file.close()
        result = {          
         'estClassTrain': est_train, 
         'predClassLabel': predClassLabel,
         'labelTest': labelTest        
        }        
        return result
              
###########################################################
def avalResult(paramClass,labelTest,lenTest,lenClass):    
    count = 0  
    predClassLabel = []
    hitsClassLabel = []
    for i in range(0,lenTest):                
        labelSeq = labelTest[i]          
        mllksClass = []        
        majorMLLLK = 0        
        for l in range(0,lenClass):
                mllksClass.append(paramClass[i][l]['mllk'])        
        majorMLLLK = np.max(mllksClass)                    
        for j in range(0,lenClass):
            nClasse = paramClass[i][j]['nClass']
            mllk = paramClass[i][j]['mllk']
            AIC = paramClass[i][j]['AIC']
            BIC = paramClass[i][j]['BIC']           
            if(mllk >= majorMLLLK):               
               predClassLabel.append(nClasse)            
            if(mllk >= majorMLLLK and labelSeq == nClasse ): 
               count = count + 1               
        accuracy = (count / lenTest)*100 
        
    result = {          
      'accuracy': accuracy, 
      'predClassLabel': predClassLabel,
      'labelTest': labelTest
    }  
    return result    
    
####################################################################
def runAlgorithm(train,test,m,est_algorithm,aval_algorithm):        
    train = baseClass(train[0]) 
    estClassTrain = []
    lenClass = len(train['classLen'])        
    for i in range(0,lenClass):        
        estClassTrain.append(est_algorithm(train['labelsParams'][i]['classParam'],m, train["base"], train['classBase'][i], train['classLen'][i]))
    test = test[0]
    labelTest = []
    paramClass = []
    lenTest = len(test['Labels'])
    for i in range(0,lenTest):    
         paramTest = []
         aux = test['Params'][i] 
         labelTest.append(test['Labels'][i])   
         for j in range(0,lenClass):
             x = []
             x.append(aux)
             paramTest.append(aval_algorithm(x,m, estClassTrain[j]['est_delta'],
                                                 estClassTrain[j]['est_gamma'],
                                                 estClassTrain[j]['est_mu'],
                                                 estClassTrain[j]['est_sigma2'],
                                                 estClassTrain[j]['nClass'],
                                                 estClassTrain[j]['nBase'],
                                                 estClassTrain[j]['minLLK'],
                                                 estClassTrain[j]['maxLLK']))         
         print("Proc....seq:", i+1, " Seq de Label",  labelTest[i])            
         paramClass.append(paramTest)     
    result = {          
      'estClassTrain': estClassTrain, 
      'paramClass': paramClass,
      'labelTest': labelTest,
      'lenTest': lenTest,
      'lenClass': lenClass
    }  
    return result   

####################################################################
def procAlgorithm(train,test,m,est_algorithm,aval_algorithm):    
    r = runAlgorithm(train,test,m,est_algorithm,aval_algorithm) 
    result = avalResult(r['paramClass'],r['labelTest'],r['lenTest'],r['lenClass'])
    estTrain = r['estClassTrain']
    accuracy = result['accuracy']    
    labelTest = result['labelTest']
    predClassLabel = result['predClassLabel']    
    analise = analiseBetterTrain(accuracy,estTrain,labelTest,predClassLabel)
    test = list(analise['labelTest'])
    pred = list(analise['predClassLabel'])    
    print("\nEnviando melhor acurácia de treino:")    
    showResults(test,pred) 
    #print("\nMelhor Treino:", analise['estClassTrain'])
    
####################################################################    
    
        
def init_hmm(train,test,m):    
    #Veross = procAlgorithm(train,test,m,estVeross,avalVeross)
    BaumWelch = procAlgorithm(train,test,m,estBaumWelch,avalBaumWelch)   
    #LogAlpha = procAlgorithm(train,test,m,estLogAlpha,avalBaumWelch)   
    #LogBeta = procAlgorithm(train,test,m,estLogBeta,avalLogBeta)   
    #LogAlphaBeta = procAlgorithm(train,test,m,estLogAlphaBeta,avalLogAlphaBeta)

     
   
