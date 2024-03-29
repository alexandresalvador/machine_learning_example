from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

acuracias = []
for x in range(0, 100):
    dataIris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)  # sep="" personalizar o separador

    # print(dataIris.head())

    labels = dataIris[4]
    data = dataIris.drop(4, axis=1)

    # print(labels.head())
    # print(data.head())

    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.2)
    # print("\ndata_train:\n")
    # print(data_train.head())
    # print(data_train.shape)

    # print("\ndata_test:\n")
    # print(data_test.head())
    # print(data_test.shape)

    modelo = MultinomialNB()
    modelo.fit(data_train, label_train)
    predicao = modelo.predict(data_test)
    # print(predicao)
    acuracia = accuracy_score(predicao, label_test)*100
    acuracias.append(acuracia)
    # print(acuracia)

print(np.mean(acuracia)) # media
# np.std - desvio padrao
# np.min -vl minimo
# np.max - maximo
# 
# 
#print(label_test)
print("Matriz de confusão")
print(confusion_matrix(predicao, label_test))
print(classification_report(label_test, predicao))
