import numpy as np
import math
import timeit
from preProcessDatabase import formatValues
from preProcessDatabase import normalizeRescale
from preProcessDatabase import normalizeRescalev
from preProcessDatabase import logsumexp
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


#instalar sudo apt-get -y install python3-sklearn




def statusProc(i, s, llk, mu, sigma2, delta, gamma, nrBase, nrClass, lenClass):
    print("Atualização na iteração:", i, "Base:",  nrBase, "Class:", nrClass, "LenClass:", lenClass) 
    #print("Base:",  nrBase, "Class: ", nrClassBase, "Len:", lenClass) 
    print("#############################################################")   
    print("Média (mu) : ", mu[0], " ", mu[1])
    print("##########################################")   
    print("Variância (sigma2) :", sigma2[0], " ", sigma2[1])
    print("##########################################")   
    print("Probab inicial (delta):\n", delta[0], " ", delta[1])
    print("##########################################")   
    print("Probab transição (gamma):\n", gamma[0][0], " ",gamma[0][1], "\n", 
                                       gamma[1][0], " ",gamma[1][1] )
    print("##########################################")  
    print("Verossimilhança: ", s, " ",  llk)
    print("##########################################")




"""
ESTIMAÇÃ0 *****************************************************


A função r.dirichlet gera um vetor aleatória da distribuição
#Dirichlet com parâmetro tetha. O parâmetro da função é:
- tetha: parâmetro da distribuição Dirichlet
"""
def r_dirichlet(tetha):
  dim = len(tetha) 
  y = []
  for i in range(0,dim):    
      y.append(np.random.gamma(shape=tetha[i], scale=1) )    
  result = y/np.sum(y)  
  return result

#Esta é a função trans.ergotica que gera uma matriz de transição ergótica e seu
#vetor de distribuição inicial estacionária associado. O parâmetro da função é:
#- m: ordem da matriz de transição e do vetor de distribuição estacionária.

def trans_ergotica(m):
  #Matriz com zeros de duas dimensões (com 1 seria  np.zeros(m)) 
  gamma = np.zeros((m,m)) 
  #Matriz identidade  
  I = np.eye(m)
  check = 0
  while check==0:
    for i in range(0,m):
      gamma[i,] = r_dirichlet(I.diagonal())    
    if np.linalg.det(gamma-I)==0:
      check = 1   
  delta = np.linalg.solve((I-gamma+1).T,np.repeat(1,m))    
  result = {          
    'gamma': gamma, 
    'delta': delta  
  }  
  return result

#MU E SIGMAS INICIAIS
#Esta é a função mu_sigma2_iniciais que gera um vetor de médias iniciais e outro
#para as variâncias. Os parâmetros da função são:
#  x: é uma lista (objeto do tipo list) de S sequências. Cada elemento da lista é
#  uma sequência observada de um HMM.
#- S: número de sequências.
#- m: ordem do modelo.
  
def mu_sigma2_iniciais(x,S,m):
    y = []
    mu = []
    sigma = []
    lenghtBase = S
    if lenghtBase==1:
      y = np.concatenate((y, x[0]), axis=0)     
    else:      
         for s in range(0,lenghtBase):            
             y = np.concatenate((y, x[s]), axis=0)       
    y = np.array(sorted(y))
    T = len(y)        
    for i in range(0,m):
        init = abs(round(((i+1)-1)*T/m + 1 ))        
        end = abs(round( (i+1)*T/m )) 
        #aux = normalizeRescale3((np.random.uniform(y[init],[end]))/10000)
        #mu.append(aux)
        mu.append((np.random.uniform(y[init],[end]))/10000)
        #sigma.append(np.random.uniform((1/4)*np.std(y)**2,4*np.std(y)**2))  
        sigma.append(np.random.uniform(0,1))
    result = {          
      'mu': mu, 
      'sigma2': sigma  
    }     
    return result
#MU E SIGMAS INICIAIS

# MATRIZ P(X) DIAGONAL
#- x: valor do ponto que se deseja calcular as probabilidades.
#- mu: vetor com as médias das densidades normais.
#- sigma2: vetor com as variâncias das densidades normais.

def matriz_P(x, mu, sigma2):  
  m = len(mu)
  P = np.zeros((m,m))
  for i in range(0,m):    
    P[i,i] = np.random.normal(mu[i],np.sqrt(sigma2[i]))                 
  return P 
# MATRIZ P(X) DIAGONAL

#VEROSSIMILHANÇA ###############################################
#- x: amostra para a qual se clacula a verossimilhança.
#- delta: vetor das probabilidades iniciais da CM associada ao modelo.
#- gamma: matriz de transição da CM associada ao modelo.
#- mu: vetor com as médias das densidades normais.
#- sigma2: vetor com as variâncias das densidades normais.
#  Veross: Lt = δΓP(x1)ΓP(x2 )...ΓP(xn )1'

def veross(x, delta, gamma, mu, sigma2):  
    m = len(delta) 
    l = len(x) 
    uns = np.ones((m, 1)) 
    Lt = delta@matriz_P(x[0], mu, sigma2)    
    for i in range(0,m):           
        Lt[i] = formatValues(Lt[i])    
    for i in range(1,l-1):    
        Lt = Lt@gamma@matriz_P(x[i], mu, sigma2)        
        for i in range(0,m):            
            Lt[i] =  formatValues(Lt[i])  
    Lt = (Lt@uns)/1000
    return Lt

#VEROSSIMILHANÇA ##############################################


#LogAlphaEstimate ####################################################

def estLogAlpha(db, m, nrBase, nrClass, lenClass):
    startTime = timeit.default_timer()    
    lenClass = len(db)
    m = m
    delta_gamma = trans_ergotica(m)
    delta = delta_gamma['delta']
    gamma = delta_gamma['gamma']
    mu_sigma2 = mu_sigma2_iniciais(db,lenClass,m)  
    mu =  mu_sigma2['mu']
    sigma2 = mu_sigma2['sigma2']
    S = len(db)    
    for i in range(0,m):            
            mu[i] =  np.transpose(formatValues(float(mu[i])))
            sigma2[i] =  formatValues(sigma2[i])       
    T = np.zeros(S)    
    for s in range(0,S):           
            T[s] = len(db[s])        
            la = [0 for _ in range(S)]        
            llk = np.zeros(S)        
            for s in range(0,S):           
                fb = logAlphaBetha(db[s], m, delta, gamma, mu, sigma2)
                la[s] = (np.array(fb['la']))/1                                              
                c = np.max(la[s][:,int(T[s] - 1)])            
                llk[s] = c+logsumexp(np.sum(np.exp(la[s][:,int(T[s] - 1)]-c)))           
                statusProc(i+1, s+1, llk[s], mu, sigma2, delta, gamma, nrBase, nrClass, lenClass)  
                sumT = np.sum(T)           
                sumllk = np.sum(llk)                
                np1 = (m - 1) + (m - 1)*m + 2*m
                AIC = -2*(sumllk - np1)             
                BIC = -2*sumllk + np1*np.log(sumT)
                endTime = timeit.default_timer()            
                tmp = endTime - startTime                    
                result = {          
                      'est_mu': mu, 
                      'est_sigma2': sigma2, 
                      'est_gamma': gamma ,  
                      'est_delta': delta,
                      'mllk': sumllk,    
                      'AIC': AIC,
                      'BIC': BIC,
                      'tmp': tmp,
                      'nBase': nrBase, 
                      'nClass': nrClass, 
                      'lClass': lenClass,
                      'minLLK': 1,
                      'maxLLK': 1                      
                }
                return  result

#LogAlphaEstimate######################################################



#LogABetaaEstimate ####################################################

def estLogBeta(db, m, nrBase, nrClass, lenClass):
    startTime = timeit.default_timer()    
    lenClass = len(db)
    m = m
    delta_gamma = trans_ergotica(m)
    delta = delta_gamma['delta']
    gamma = delta_gamma['gamma']
    mu_sigma2 = mu_sigma2_iniciais(db,lenClass,m)  
    mu =  mu_sigma2['mu']
    sigma2 = mu_sigma2['sigma2']
    S = len(db)    
    for i in range(0,m):            
            mu[i] =  np.transpose(formatValues(float(mu[i])))
            sigma2[i] =  formatValues(sigma2[i])       
    T = np.zeros(S)    
    for s in range(0,S):           
            T[s] = len(db[s])        
            lb = [0 for _ in range(S)]        
            llk = np.zeros(S)        
            for s in range(0,S):           
                fb = logAlphaBetha(db[s], m, delta, gamma, mu, sigma2)
                lb[s] = (np.array(fb['lb']))/1                
                c = np.max(lb[s][:,int(0)])                                
                llk[s] = c+logsumexp(np.sum(np.exp(lb[s][:,int(0)]-c)))               
                statusProc(i+1, s+1, llk[s], mu, sigma2, delta, gamma, nrBase, nrClass, lenClass)
                sumT = np.sum(T)           
                sumllk = np.sum(llk)                
                np1 = (m - 1) + (m - 1)*m + 2*m
                AIC = -2*(sumllk - np1)             
                BIC = -2*sumllk + np1*np.log(sumT)
                endTime = timeit.default_timer()            
                tmp = endTime - startTime                    
                result = {          
                      'est_mu': mu, 
                      'est_sigma2': sigma2, 
                      'est_gamma': gamma ,  
                      'est_delta': delta,
                      'mllk': sumllk,    
                      'AIC': AIC,
                      'BIC': BIC,
                      'tmp': tmp,
                      'nBase': nrBase, 
                      'nClass': nrClass, 
                      'lClass': lenClass,
                      'minLLK': 1,
                      'maxLLK': 1,
                      'itr': 1                      
                }
                return  result

#LogBetaEstimate######################################################

#LogAlphaBetaEstimate ####################################################

def estLogAlphaBeta(db, m, nrBase, nrClass, lenClass):
    startTime = timeit.default_timer()    
    lenClass = len(db)
    m = m
    delta_gamma = trans_ergotica(m)
    delta = delta_gamma['delta']
    gamma = delta_gamma['gamma']
    mu_sigma2 = mu_sigma2_iniciais(db,lenClass,m)  
    mu =  mu_sigma2['mu']
    sigma2 = mu_sigma2['sigma2']
    S = len(db)    
    for i in range(0,m):            
            mu[i] =  np.transpose(formatValues(float(mu[i])))
            sigma2[i] =  formatValues(sigma2[i])       
    T = np.zeros(S)    
    for s in range(0,S):           
       T[s] = len(db[s])              
       la = [0 for _ in range(S)]
       lb = [0 for _ in range(S)]
       llk = np.zeros(S) 
       pInitial = []
       for s in range(0,S):
           fb = logAlphaBetha(db[s], m, delta, gamma, mu, sigma2)
           la[s] = (np.array(fb['la']))/1
           lb[s] = (np.array(fb['lb']))/1           
           sumLA = np.sum(la[s][:,int(T[s] - 1)]) 
           #pInitial.append( ( np.exp(la[s][:,0])@np.exp(lb[s][:,0]) ) / sumLA )  
           pInitial.append(((np.exp(la[s][:,0]) / (1 +  np.exp(la[s][:,0]))) @ 
            (np.exp(lb[s][:,0]) / (1 +  np.exp(lb[s][:,0])))) / sumLA )                                           
           #c = np.max(la[s][:,int(T[s] - 1)])                       
           #llk[s] = c+logsumexp(np.sum(np.exp(pInitial[s]-c)))
           llk[s] = np.sum(pInitial[s])
           #llk[s] = logsumexp(np.sum(np.exp(pInitial[s]))) 
           statusProc(i+1, s+1, llk[s], mu, sigma2, delta, gamma, nrBase, nrClass, lenClass)   
       sumT = np.sum(T)           
       sumllk = np.sum(llk)                
       np1 = (m - 1) + (m - 1)*m + 2*m
       AIC = -2*(sumllk - np1)             
       BIC = -2*sumllk + np1*np.log(sumT)
       endTime = timeit.default_timer()            
       tmp = endTime - startTime                    
       result = {          
                      'est_mu': mu, 
                      'est_sigma2': sigma2, 
                      'est_gamma': gamma ,  
                      'est_delta': delta,
                      'mllk': sumllk,    
                      'AIC': AIC,
                      'BIC': BIC,
                      'tmp': tmp,
                      'nBase': nrBase, 
                      'nClass': nrClass, 
                      'lClass': lenClass,
                      'minLLK': 1,
                      'maxLLK': 1,
                      'itr': 1                      
                }
       return  result

#LogAlphaBetaEstimate######################################################


#Verossestimates ###############################################################

def estVeross(classe,m,nrBase, nrClass, lenClass):
    startTime = timeit.default_timer() 
    classe = classe
    verossEstimate = []
    lenClass = len(classe)
    m = m
    delta_gamma = trans_ergotica(m)
    delta = delta_gamma['delta']
    gamma = delta_gamma['gamma']
    mu_sigma2 = mu_sigma2_iniciais(classe,lenClass,m)  
    mu =  mu_sigma2['mu']
    sigma2 = mu_sigma2['sigma2']            
    for x in range(0, lenClass):
        verossEstimate.append(veross(classe[x] , delta, gamma, mu, sigma2))
    minLLK =  np.min(verossEstimate)
    maxLLK =  np.max(verossEstimate) 
    llk = np.mean(normalizeRescale(verossEstimate))
    statusProc(1, 1, llk, mu, sigma2, delta, gamma, nrBase, nrClass, lenClass)    
    endTime = timeit.default_timer()            
    tmp = endTime - startTime   
    result = {          
          'est_mu': mu, 
          'est_sigma2': sigma2, 
          'est_gamma': gamma ,  
          'est_delta': delta,
          'mllk': llk,         
          'tmp': tmp,
          'nBase': nrBase, 
          'nClass': nrClass, 
          'lClass': lenClass,
          'minLLK': minLLK,
          'maxLLK': maxLLK
    }
    return  result

#LOG - FORWARD E BACKWARD ###################################
#- x: sequência observada de um HMM.
#- m: ordem do modelo.
#- gamma: matriz quadrada m × m. Matriz de transição da cadeia de Markov.
#- delta: vetor de tamanho m. Distribuição incial da cadeia de Markov.
#- mu: vetor de tamanho m, que guarda as médias das distribuições normais.
#- sigma2: vetor de tamanho m, que guarda as variâncias das distribuições nor-
#mais.
#αlpha = δP(x1)ΓP(x2 ) . . . ΓP(xt )
#betha = ΓP(xt+1 )ΓP(xt+2 ) . . . ΓP(xT )1'

def logAlphaBetha(x, m, delta, gamma, mu, sigma2):        
    T = len(x) 
    lalpha = np.zeros((m,T))
    lbeta = np.zeros((m,T)) 
    allprobs = np.zeros((T,m))  
    div = 10000
    for t in range(0,T):           
        P = matriz_P(x[t], mu, sigma2)
        allprobs[t,] = np.diag(P)        
    foo1 = delta*allprobs[0]     
    sumfoo1 = np.sum(foo1)      
    lscale1 =  logsumexp(sumfoo1)              
    foo1 = foo1 / sumfoo1  
    lalpha[:,0] = logsumexp(foo1) + lscale1  
    for t in range(1,T):                 
        foo1 = foo1@gamma*allprobs[t,]           
        sumfoo1 = np.sum(foo1)                  
        lscale1 = lscale1 + logsumexp(sumfoo1)                 
        foo1 = (foo1) / (sumfoo1)                    
        lalpha[:,t] = (logsumexp(foo1) + lscale1)/div                                 
    lbeta[:,T - 1] = (np.repeat(0,m))    
    foo2 = np.repeat(1/m, m)
    lscale2 =  logsumexp(np.float64(m))     
    for t in range(1,T):       
        foo2 = gamma@(allprobs[(T - t),]*foo2)                              
        lbeta[:,(T - t) - 1 ] = ((logsumexp(foo2) + lscale2))/div       
        sumfoo2 = np.sum(foo2)        
        foo2 = (foo2/sumfoo2)                         
        lscale2 = lscale2 + logsumexp(sumfoo2)
    #print("lbeta", lbeta)            
    result = {          
      'la': lalpha, 
      'lb': lbeta       
    }
    return result
#LOG - FORWARD E BACKWARD ###################################

"""
Aqui a função BaumWelch que calcula o algoritmo Baumm-Welch para S
sequências x observadas. Retornando as estimativas dos parâmetros do modelo, a
verossimilhança máxima obtida, os valores dos critérios AIC e BIC e o número de
iterações necessárias para a convergência.
Os parâmetros da função são:
- x: é uma lista (objeto do tipo list) de S sequências. Cada elemento da lista é
uma sequência observada de um HMM.
- m: ordem do modelo.
- delta: vetor de tamanho m. Parâmetro inicial para a distribuição incial da
cadeia de Markov.
- gamma: matriz quadrada m × m. Parâmetro incial para a matriz de transição
da cadeia de Markov.
- mu: vetor de tamanho m, que guarda os parâmetros iniciais das médias das
distribuições normais.
- sigma2: vetor de tamanho m, que guarda os parâmetros iniciais das variâncias
das distribuições normais.
- I: número máximo de iterações, aqui sugere-se I = 1000.
- criterio: critério que indica a convergência do algoritmo, aqui sugere-se criterio=
10( − 6).

"""

#TESTE 2 ########################################
def BaumWelch(db, S, m, delta, gamma, mu, sigma2, I, ct, nrBase, nrClass, lenClass):    
    for i in range(0,m):            
        mu[i] =  np.transpose(formatValues(float(mu[i])))
        sigma2[i] =  formatValues(sigma2[i])   
    mu_next = mu
    sigma2_next = sigma2
    delta_next = delta
    gamma_next = gamma    
    history_data = [0 for _ in range(I)]
    history_crit = [0 for _ in range(I)]
    #ct = 0.20          
    T = np.zeros(S)    
    for s in range(0,S):           
        T[s] = len(db[s])        
    for it in range(0,I):        
        lallprobs = list([0 for _ in range(S)])      
        la = [0 for _ in range(S)]
        lb = [0 for _ in range(S)]
        llk = np.zeros(S)   
        pInitial = []
        for s in range(0,S):
            probs = np.zeros((int(T[s]),m))                        
            for t in range(0,int(T[s])):                
                P = matriz_P(db[s][t], mu, sigma2)                 
                probs[t,] = logsumexp(np.diag(P))                       
            lallprobs[s] = np.array(probs)/10000                           
            fb = logAlphaBetha(db[s], m, delta, gamma, mu, sigma2)
            la[s] = (np.array(fb['la']))/1
            lb[s] = (np.array(fb['lb']))/1
            sumLA = np.sum(la[s][:,int(T[s] - 1)])              
            #pInitial.append( ( np.log(la[s][:,0]) * np.log(lb[s][:,0]) ) / sumLA )            
            pInitial.append(((np.exp(la[s][:,0]) / (1 +  np.exp(la[s][:,0]))) * 
                              (np.exp(lb[s][:,0]) / (1 +  np.exp(lb[s][:,0])))) / sumLA )                                           
            c = np.max(la[s][:,int(T[s] - 1)])             
            llk[s] = c+logsumexp(np.sum(np.exp(la[s][:,int(T[s] - 1)]-c)))
            statusProc(it+1, s+1, llk[s], mu, sigma2, delta, gamma, nrBase, nrClass, lenClass)                
        #u1 = np.repeat(0,m)        
        F = np.zeros((m,m))            
        for s in range(0,S):
            for j in range(0,m):
                  for k in range(0,m):
                      #F[j,k] = F[j,k] + gamma[j,k]*sum(np.exp(la[s][j:int(T[s])-1]))
                      a = la[s][j,0:int(T[s]-1)]                                            
                      b = lallprobs[s][1:int(T[s]),k]                      
                      c = lb[s][k,1:int(T[s])] - llk[s]
                      F[j,k] = (F[j,k] + gamma[j,k] * np.sum(np.exp(a + b + c)))/1000         
            #u1 = u1 + np.exp( la[s][:,0] + lb[s][:,0] - llk[s] )        
        aux = pInitial[0]
        x = aux[0]
        if(math.isnan(x)):
            x = 0.5    
            it = 0
        scale = int(len(str(round(x,0)))) -2        
        aux1 = np.abs(round(x / 10 ** scale,15))
        u1 = np.repeat(aux1,m)        
        delta_next = u1       
        gamma_next = F / np.transpose(np.transpose(np.apply_over_axes(lambda F, axis: np.sum(F, axis=axis), F, axes=(1))))
        for j in range(0,m):
            aux_div = 0
            aux_mu = 0
            aux_sigma2 = 0
            for s in range(0,S):
               aux_div = aux_div + np.sum(np.exp(la[s][j,] + lb[s][j,] - llk[s]))                              
               aux_mu  = aux_mu + np.sum( np.exp(la[s][j,] + lb[s][j,] - llk[s]) * db[s]  )                             
            mu_next[j] =  aux_mu / aux_div                         
            for s in range(0,S):
                aux_sigma2  = aux_sigma2 + np.sum( np.exp(la[s][j,] + lb[s][j,] - llk[s]) * (db[s] - mu_next[j])**2  )                          
            sigma2_next[j] = aux_sigma2 / aux_div         
        for i in range(0,m):            
            mu_next[i] =  np.transpose(formatValues(float(mu[i])))
            sigma2_next[i] =  formatValues(sigma2[i])
        #Analysis criteria
        crit1 = np.sum(np.abs(np.array(mu) - np.array(mu_next))) 
        crit2 = np.sum(np.abs(np.array(sigma2) - np.array(sigma2_next))) 
        crit3 = np.sum(np.abs(np.array(gamma) - np.array(gamma_next))) 
        crit4 = np.sum(np.abs(np.array(delta) - np.array(delta_next)))
        crit = crit1 + crit2 + crit3 + crit4        
        #History
        minLLK = np.min(llk) 
        maxLLK = np.max(llk)  
        sumT = np.sum(T)                
        sumllk = np.sum(normalizeRescale(llk))                 
        np1 = (m - 1) + (m - 1)*m + 2*m
        AIC = -2*(sumllk - np1)                      
        BIC = -2*sumllk + np1*np.log(sumT)        
        itData = { 
          'crit': crit,
          'mu': mu, 
          'sigma2': sigma2, 
          'gamma': gamma,  
          'delta': delta,
          'mllk': sumllk,
          'AIC': AIC,
          'BIC': BIC,
          'iter': it+1,
          'minLLK': minLLK,
          'maxLLK': maxLLK
        }        
        history_data[it] = itData
        history_crit[it] = crit
        #History END                                
        if(math.isnan(crit)):
            print("deu zebra")             
            return 0             
        else:            
            if(crit < ct or it == I):
                minLLK = np.min(llk) 
                maxLLK = np.max(llk)  
                sumT = np.sum(T)                
                sumllk = np.sum(normalizeRescale(llk))                 
                np1 = (m - 1) + (m - 1)*m + 2*m
                AIC = -2*(sumllk - np1)                      
                BIC = -2*sumllk + np1*np.log(sumT)   
                result = {          
                  'mu': mu, 
                  'sigma2': sigma2, 
                  'gamma': gamma,  
                  'delta': delta,
                  'mllk': sumllk,
                  'AIC': AIC,
                  'BIC': BIC,
                  'iter': it+1,
                  'minLLK': minLLK,
                  'maxLLK': maxLLK
                }
                print("#########################################")
                print("O algoritmo convergiu depois de", it+1, "Iterações")               
                print("#########################################")
                return result                
            mu = mu_next
            sigma2 = sigma2_next
            gamma = gamma_next
            delta = delta_next
    crit_menor = np.min(history_crit)
    crit_major = np.max(history_crit)        
    print("O algoritmo não convergiu depois de ", I, "Iterações", 
          "\nMenor convergência de:" , crit_major, 
          "\nEnviando maior convergência:", crit_menor)    
    for i in range(0,I):
       if (crit_menor == history_data[i]['crit'] ):                       
            result = {          
              'mu': history_data[i]['mu'], 
              'sigma2': history_data[i]['sigma2'], 
              'gamma': history_data[i]['gamma'],  
              'delta': history_data[i]['delta'],
              'mllk': history_data[i]['mllk'],
              'AIC': history_data[i]['AIC'],
              'BIC': history_data[i]['BIC'],
              'iter': history_data[i]['iter'],
              'minLLK': history_data[i]['minLLK'],
              'maxLLK': history_data[i]['maxLLK']
            }           
    return result 
#ALGORITMO BaumWelch END  ############

##Avalia BaumWelch
# A função estBaumWelch inicializa o algoritmo ExpMax com N
#diferentes conjuntos de pontos iniciais, retornando a estimativa que apresenta a maior
#das verossimilhanças.
#Os parâmetros da função são:
#- x: é uma lista (objeto do tipo list) de S sequências. Cada elemento da lista é
#uma sequência observada de um HMM.
#- S: número de sequências observadas.
#- m: ordem do modelo.
def estBaumWelch(database,m,nrBase, nrClassBase, lenClass):
    startTime = timeit.default_timer()
    db = database
    S = len(db)
    m = m        
    I = 1
    #I = 2
    N = 50
    crit = 10**(-6) 
    print("***************************")       
    estimation_mu = [None for _ in range(m)]
    estimation_sigma2 = [None for _ in range(m)]
    estimation_delta = [None for _ in range(m)]
    estimation_gamma = np.zeros((m,m)) 
    AIC = 0
    BIC = 0    
    mllk = 0
    itr = 0
    hist_mllk = np.zeros(N) 
    iter_necessary= np.zeros(N) 
    betterInitial_mu = np.zeros(m) 
    betterInitial_sigma2 = np.zeros(m) 
    betterInitial_delta = np.zeros(m) 
    betterInitial_gamma = np.zeros((m,m)) 
    n = 1
    while (n < N+1):
        print("iteração da avaliação", n)        
        m_s2 = mu_sigma2_iniciais(db,S,m)
        mu = m_s2['mu']
        sigma2 = m_s2['sigma2']
        gamma_delta = trans_ergotica(m)
        gamma = gamma_delta['gamma']
        delta = gamma_delta['delta']    
        result = BaumWelch(db,S,m,delta,gamma,mu,sigma2,I,crit,nrBase, nrClassBase, lenClass)
        if (result == 0):
            r = 0
            ###################################
            mu = m_s2['mu']
            sigma2 = m_s2['sigma2']
            gamma_delta = trans_ergotica(m)
            gamma = gamma_delta['gamma']
            delta = gamma_delta['delta']    
            result = BaumWelch(db,S,m,delta,gamma,mu,sigma2,I,crit,nrBase, nrClassBase, lenClass)
            ##################################E
        else:
            r = int(str(len(result.keys())))        
        if(r < 2 ):
            n = n
        else:
            hist_mllk[n] = result['mllk']            
            iter_necessary[n] = result['iter']
            if(n == 1):
                ###############################
                if(result['mllk'] == 0.0):
                    mu = m_s2['mu']
                    sigma2 = m_s2['sigma2']
                    gamma_delta = trans_ergotica(m)
                    gamma = gamma_delta['gamma']
                    delta = gamma_delta['delta']    
                    result = BaumWelch(db,S,m,delta,gamma,mu,sigma2,I,crit,nrBase, nrClassBase, lenClass)
                else:                    
                ###############################
                    estimation_mu = result['mu']
                    estimation_sigma2 = result['sigma2']
                    estimation_delta = result['delta']
                    estimation_gamma = result['gamma']
                    AIC = result['AIC']
                    BIC = result['BIC']
                    itr = result['iter']
                    maxLLK = result['maxLLK']
                    minLLK = result['minLLK'] 
                    betterInitial_mu = mu
                    betterInitial_sigma2 = sigma2
                    betterInitial_delta = delta
                    betterInitial_gamma = gamma                
                    mllk = np.max(hist_mllk) 
            else:
                if(hist_mllk[n]>hist_mllk[n-1]):
                    ###############################
                    if(result['mllk'] == 0.0):
                        mu = m_s2['mu']
                        sigma2 = m_s2['sigma2']
                        gamma_delta = trans_ergotica(m)
                        gamma = gamma_delta['gamma']
                        delta = gamma_delta['delta']    
                        result = BaumWelch(db,S,m,delta,gamma,mu,sigma2,I,crit,nrBase, nrClassBase, lenClass)
                    else:                    
                    ###############################                                        
                        estimation_mu = result['mu']
                        estimation_sigma2 = result['sigma2']
                        estimation_delta = result['delta']
                        estimation_gamma = result['gamma']
                        AIC = result['AIC']
                        BIC = result['BIC']
                        itr = result['iter']
                        maxLLK = result['maxLLK']
                        minLLK = result['minLLK']                       
                        betterInitial_mu = mu
                        betterInitial_sigma2 = sigma2
                        betterInitial_delta = delta
                        betterInitial_gamma = gamma                    
                        mllk = np.max(hist_mllk)                     
            n = n+1
            endTime = timeit.default_timer()            
            tmp = endTime - startTime                       
        result = {          
          'est_mu': estimation_mu, 
          'est_sigma2': estimation_sigma2, 
          'est_gamma': estimation_gamma ,  
          'est_delta': estimation_delta,
          'mllk': mllk,
          'AIC': AIC,
          'BIC': BIC,
          'itr': itr,
          'tmp': tmp,
          'nBase': nrBase, 
          'nClass': nrClassBase, 
          'lClass': lenClass,
          'minLLK': minLLK,
          'maxLLK': maxLLK
                          
        }                
        return result     

"""
ESTIMAÇÃ0  FIM *****************************************************
"""
###################################################################

"""SELEÇÃ0  INÍCIO *****************************************************
"""
def avalBaumWelch(db, m, delta, gamma, mu, sigma2, nrClass, nrBase, minLLK, maxLLK):
    S = len(db)    
    for i in range(0,m):            
        mu[i] =  np.transpose(formatValues(float(mu[i])))
        sigma2[i] =  formatValues(sigma2[i])   
    mu = mu
    sigma2 = sigma2
    delta = delta
    gamma = gamma           
    T = np.zeros(S)    
    for s in range(0,S):           
        T[s] = len(db[s])        
        la = [0 for _ in range(S)]        
        llk = np.zeros(S)        
        for s in range(0,S):           
            fb = logAlphaBetha(db[s], m, delta, gamma, mu, sigma2)
            la[s] = (np.array(fb['la']))/1                                              
            c = np.max(la[s][:,int(T[s] - 1)])            
            llk[s] = c+logsumexp(np.sum(np.exp(la[s][:,int(T[s] - 1)]-c)))            
            sumT = np.sum(T)           
            sumllk = np.sum(llk)                
            np1 = (m - 1) + (m - 1)*m + 2*m
            AIC = -2*(sumllk - np1)             
            BIC = -2*sumllk + np1*np.log(sumT)                    
            result = { 
                  'nClass': nrClass,
                  'mllk': sumllk,
                  'AIC': AIC,
                  'BIC': BIC,  
                  'nBase': nrBase                  
                }
            #print("Sequencia analisada.....")               
            
            return result               
###################################################################

def avalVeross(classe,m,delta, gamma, mu, sigma2, nrClass, nrBase, minLLK, maxLLK):
    startTime = timeit.default_timer() 
    classe = classe
    T = len(classe)
    verossEstimate = []   
    m = m    
    delta = delta
    gamma = gamma      
    mu =  mu
    sigma2 = mu       
    #print("***************************")         
    for x in range(0, T):
        verossEstimate.append(veross(classe[x], delta, gamma, mu, sigma2))
    minLLK =  minLLK
    maxLLK =  maxLLK     
    llk = verossEstimate
    endTime = timeit.default_timer()            
    time = endTime - startTime   
    sumT = np.sum(T)           
    sumllk = np.sum(llk)                
    np1 = (m - 1) + (m - 1)*m + 2*m
    AIC = -2*(sumllk - np1)             
    BIC = -2*sumllk + np1*np.log(sumT)                    
    result = {                 
          'nClass': nrClass,
          'mllk': sumllk,
          'AIC': AIC,
          'BIC': BIC,  
          'nBase': nrBase,
          'time': time                  
        } 
    return result

def avalLogAlphaBeta(db, m, delta, gamma, mu, sigma2, nrClass, nrBase, minLLK, maxLLK):
       startTime = timeit.default_timer()    
       lenClass = len(db)
       m = m
       delta = delta      
       gamma = gamma
       sigma2 =sigma2
       mu =  mu       
       S = len(db)    
       for i in range(0,m):            
               mu[i] =  np.transpose(formatValues(float(mu[i])))
               sigma2[i] =  formatValues(sigma2[i])       
       T = np.zeros(S)    
       for s in range(0,S):           
          T[s] = len(db[s])              
          la = [0 for _ in range(S)]
          lb = [0 for _ in range(S)]
          llk = np.zeros(S) 
          pInitial = []
          for s in range(0,S):
              fb = logAlphaBetha(db[s], m, delta, gamma, mu, sigma2)
              la[s] = (np.array(fb['la']))/1
              lb[s] = (np.array(fb['lb']))/1           
              sumLA = np.sum(la[s][:,int(T[s] - 1)]) 
              #pInitial.append( ( np.exp(la[s][:,0])@np.exp(lb[s][:,0]) ) / sumLA )  
              pInitial.append(((np.exp(la[s][:,0]) / (1 +  np.exp(la[s][:,0]))) @
                                (np.exp(lb[s][:,0]) / (1 +  np.exp(lb[s][:,0])))) / sumLA )                                           
              c = np.max(la[s][:,int(T[s] - 1)]) 
              llk[s] = np.sum(pInitial[s])
              #llk[s] = logsumexp(np.sum(np.exp(pInitial[s])))               
              #llk[s] = c+logsumexp(np.sum(np.exp(pInitial[s]-c)))               
          sumT = np.sum(T)           
          sumllk = np.sum(llk)                
          np1 = (m - 1) + (m - 1)*m + 2*m
          AIC = -2*(sumllk - np1)             
          BIC = -2*sumllk + np1*np.log(sumT)
          endTime = timeit.default_timer()            
          tmp = endTime - startTime                    
          result = {          
                         'est_mu': mu, 
                         'est_sigma2': sigma2, 
                         'est_gamma': gamma ,  
                         'est_delta': delta,
                         'mllk': sumllk,    
                         'AIC': AIC,
                         'BIC': BIC,
                         'tmp': tmp,
                         'nBase': nrBase, 
                         'nClass': nrClass, 
                         'lClass': lenClass,
                         'minLLK': 1,
                         'maxLLK': 1,
                         'itr': 1                      
                   }
          return  result
        
def avalLogBeta(db, m, delta, gamma, mu, sigma2, nrClass, nrBase, minLLK, maxLLK):
    S = len(db)    
    for i in range(0,m):            
        mu[i] =  np.transpose(formatValues(float(mu[i])))
        sigma2[i] =  formatValues(sigma2[i])   
    mu = mu
    sigma2 = sigma2
    delta = delta
    gamma = gamma           
    T = np.zeros(S)    
    for s in range(0,S):           
        T[s] = len(db[s])        
        lb = [0 for _ in range(S)]        
        llk = np.zeros(S)        
        for s in range(0,S):           
            fb = logAlphaBetha(db[s], m, delta, gamma, mu, sigma2)
            lb[s] = (np.array(fb['lb']))/1                                              
            c = np.max(lb[s][:,int(0)])            
            llk[s] = c+logsumexp(np.sum(np.exp(lb[s][:,int(0)]-c)))            
            sumT = np.sum(T)           
            sumllk = np.sum(llk)                
            np1 = (m - 1) + (m - 1)*m + 2*m
            AIC = -2*(sumllk - np1)             
            BIC = -2*sumllk + np1*np.log(sumT)                    
            result = { 
                  'nClass': nrClass,
                  'mllk': sumllk,
                  'AIC': AIC,
                  'BIC': BIC,  
                  'nBase': nrBase                  
                }
            #print("Sequencia analisada.....")               
            
            return result               
######################### Viterbi ######################################

# define Viterbi algorithm for shortest path
# code adapted from Stephen Marsland's, Machine Learning An Algorthmic Perspective, Vol. 2
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

    path, delta, phi = viterbi(pi, a, b, obs)
    print('\nsingle best state path: \n', path)
    print('delta:\n', delta)
    print('phi:\n', phi)


######################### Viterbi #################################

"""SELEÇÃ0  FIM *****************************************************
"""










   
