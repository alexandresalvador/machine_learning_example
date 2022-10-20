import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
# como ler e pegar as informaçoes
datairis = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
header=None) # sep="" personalizar o separador


conjuntoTreinamento = datairis.sample(60)
labels_treinamento = np.array(conjuntoTreinamento.get(4))
dados_treinamento = np.array(conjuntoTreinamento)[:, :4]

#print(dados_treinamento)
#print(labels_treinamento)

modelo = MultinomialNB()
modelo.fit(dados_treinamento, labels_treinamento)

teste = [
   [ 4.9, 3.7, 5.1, 0.1],
   [ 5.8, 2.8, 1.5, 1.9]
]

print(modelo.predict(teste))