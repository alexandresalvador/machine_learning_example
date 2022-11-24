import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model as lm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acuracias = []

for x in range(0, 2):

    dataWine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
    print(dataWine.head())

    labels = dataWine[0]
    data = dataWine.drop(0, axis=1)

    data_train, data_test, label_train, label_test = train_test_split(
    data, labels, test_size=0.2)

    model = lm.LinearRegression()   
    model.fit(data_train, label_train)
    predicao = model.predict(data_test)


    acuracia = accuracy_score
    (predicao, label_test) * 100
    acuracias.append(acuracia)
    print("Acurácia: ", acuracia)

print('med: ', np.mean( acuracia ))

print('min: ', np.min( acuracias ))

print('max: ', np.max( acuracias ))

dados01 = [[13.71,1.86,2.36,16.6,101,2.61,2.88,.27,1.69,3.8,1.11,4,1035]] #1

dados02 = [[12,1.51,2.42,22,86,1.45,1.25,.5,1.63,3.6,1.05,2.65,450]] #2

dados03 = [[13.17,5.19,2.32,22,93,1.74,.63,.61,1.55,7.9,.6,1.48,725]] #3

predicao01 = model.predict(dados01)
predicao02 = model.predict(dados02)
predicao03 = model.predict(dados03)


print('Prediction 01: ', predicao01)
print('Prediction 02: ', predicao02)
print('Prediction 03: ', predicao03)

print("Matriz de confusão")
print(confusion_matrix(predicao, label_test))
print(classification_report(label_test, predicao))
