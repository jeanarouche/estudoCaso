import numpy as np
import json
def treatLine(x):
    line = []    
    for i in range(0,len(x)): 
        line.append(float(x[i]))   
        #line.append(x[i]) 
    return line

def treatDatabase(x):     
    Labels = []
    Params = []
    Database = []    
    for i in range(0,len(x)): 
        #Database.append(treatLine(x[i].split(',')))
        Database.append(treatLine(x[i]))
        Labels.append(Database[i][0])
        #Aqui treinamento não supervisionado excluindo o ŕotulou
        #Params.append(Database[i][1:len(Database[i])])
        #Aqui treinamento supervisionado incluindo o ŕotulou
        Params.append(Database[i][1:len(Database[i])])  
        print("Base:" + repr(len(x)) +"-Line:" + repr(i + 1)  + "-Label:" + repr(Database[i][0]) + "-seqTam:" + repr(len(Params[i])))  
        
    result = {
    'Labels': Labels,
    'Params': Params
    }    
    return  result 

def treatDatabase2(x,y):     
    Labels = y
    Params = x
        
    result = {
    'Labels': Labels,
    'Params': Params
    }    
    return  result 

 

def formatValues(v):       
        aux = "" + repr(v)
        v = np.float64(aux[:8])        
        return v    
    
def normalizeRescalev(v,vmin,vmax):       
        highestValue = vmax       
        lowerValue =   vmin        
        minValue = 0
        maxValue = 1            
        newValue = minValue + (((v - lowerValue)/(highestValue - lowerValue)) * (maxValue - minValue) )            
        return newValue
    
def normalizeRescale(v):           
        normalizeValues = []
        highestValue = max(v)       
        lowerValue =   min(v)         
        minValue = 0
        maxValue = 1
        for i in range(0,len(v)):     
            newValue = minValue + ( ((v[i] - lowerValue)/(highestValue - lowerValue)) * (maxValue - minValue) )
            normalizeValues.append(newValue)
            #print("Linha: " + repr(i+1) + " Valor novo: " + repr(newValue[0]))        
        return normalizeValues

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def classLabel(basesTrain):    
    #print("tam", T)
    for i in range(0,len(basesTrain)):        
        aux = basesTrain['Labels']
        r = []
        [r.append(i) for i in aux if not r.count(i)]
        classLabel  = sorted(r)
    return classLabel 

def separateClass(basesTrain, c):
    
    classRec = c  
    classParams = []
    classLabels = [] 
    T = len(basesTrain['Labels'])    
    for i in range(0,T):          
        if (classRec == basesTrain['Labels'][i] ):                    
            classLabels.append(basesTrain['Labels'][i])
            classParams.append(basesTrain['Params'][i])  
    #print("Base", T, "Classe", c  ," tamClasse", len(classLabels))
    classLen = len(classLabels)
    result = {          
      'classLabel': classLabels, 
      'classParam': classParams, 
      'classLen': classLen  
    }     
    return result

def baseClass(basesTrain):
    labelsParams = []    
    T = len(basesTrain['Labels'])      
    classBase = classLabel(basesTrain) 
    classLen = []                       
    for j in range(0,len(classBase)):                      
            labelsParams.append(separateClass(basesTrain, classBase[j]))
            classLen.append(len(labelsParams[j]['classLabel']))    
            
    result = {          
      'labelsParams': labelsParams, 
      'base': T,
      'classBase' : classBase,
      'classLen' : classLen
    }     
    return result

def treatHistoryTrain(data):    
   
    characters = "()"
    for x in range(len(characters)):
        data = data.replace(characters[x], "")    
    data = data.replace("array", "")
    data = data.replace("\n", "")
    data = data.replace("'", '''"''')     
    # printing initial json
    ini_string = json.dumps(data)      
    # converting string to json
    final_dictionary = json.loads(ini_string)
    est_train = str(final_dictionary)
    return est_train





