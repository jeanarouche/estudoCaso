Algoritmo HMM para estudo de caso

1) Estrutura da codificação
1.1) main.py = arquivo responsável em iniciar a aplicação 
1.2) mainHMM = arquivo responsável em iniciar o algoritmo HMM inserindo dados de treino e teste para fazer as estimativas, classificação e avaliação dos dados. E também pode-se utilizar 5 formas diferentes para estimação de parâmetros do algoritmo HMM com a VEROSSIMILHANÇA, a probabilidade FORWARD, a probabilidade BACKWARD, a probabilidade conjunta BACKWARD-FORWARD e o algoritmo BAUM-WELCH. E para variar entre as bases de treino utiliza valores entre 0 a 3 na variável train = baseClass(train[0]) da função runAlgorithm. 
-Variar bases =  para variar entre as bases de treino utiliza valores entre 0 a 3 na variável train = baseClass(train[0]) da função runAlgorithm.
-Variar tipos de HMM  = utilizar a função “init_hmm”, pelo qual pode-se comentar ou descomentar: Veross, BaumWelch, LogAlpha, LogBeta, LogAlphaBeta 
1.3) loadDataSet.py = carrega as bases de dados de todo DataSet. Neste arquivo existe a variável "pathDatabase" que se deve colocar o caminho absoluto para encontrar o diretório onde estão localizados os arquivos dos dados.
1.4) preProcessDatabase.py = faz o pré-processamento das bases de dados. 
1.5) Diretório history = deve-se criar esse diretório no mesmo diretório dos outros arquivos. Nele estarão localizados os arquivos accuracyLast.json (guarda a melhor acurácia de treino), est_train.json (guarda valores dos parâmetros do HMM da melhor acurácia de treino), labelTest.json (guarda rótulos de teste da melhor acurácia estimada) e predClassLabel.json (guarda rótulos de treino da melhor acurácia estimada).

2) As bases de dados foram disponibilizadas com os seguintes nomes: 
- Classes.npy
- Dados_1.npy
- Dados_2.npy
- Dados_3.npy
- Dados_4.npy
- Dados_5.npy

 
