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


print("Matriz de confusão")
print(confusion_matrix(predicao, label_test))
print(classification_report(label_test, predicao))
