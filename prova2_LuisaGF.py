import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#PASSO 1: IMPORTAÇÃO E SEPARAÇÃO DOS DADOS

base = pd.read_csv('heart.csv')
print(base)

previsores = base.iloc[:, 0:13].values
classe = base.iloc[:, 13].values
print("previsores:", previsores)
print("classe: ", classe)

def execute():

    # PASSO 2: NORMALIZAÇÃO DA BASE DE DADOS
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    previsores_normalizado = scaler.fit_transform(previsores)

    # PASSO 3: VALIDAÇÃO/ESCOLHA DO MELHOR VALOR PARA K

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    error1 = []
    for i in range(3,10,2):
        knn1 =KNeighborsClassifier(n_neighbors=i)
        knn1.fit(previsores, classe)
        res1 = knn1.predict(previsores)
        pred_i = knn1.predict(previsores)
        error1.append(np.mean(pred_i != classe))
        print('Acurácia do modelo sem normalização para k={} '.format(i), accuracy_score(classe, res1))

    error2 = []
    for y in range (3,10,2):
        knn2 = KNeighborsClassifier(n_neighbors=y)
        knn2.fit(previsores_normalizado, classe)
        res2 = knn2.predict(previsores_normalizado)
        pred_y = knn2.predict(previsores_normalizado)
        error2.append(np.mean(pred_y != classe))
        print('Acurácia do modelo com normalização para k={} '.format(y), accuracy_score(classe, res2))

    plt.figure(figsize=(12, 6))
    plt.plot(range(3,10,2), error1, color='magenta', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Taxa de erro(sem normalização) X Valores de K')
    plt.xlabel('Valores para k')
    plt.ylabel('Erro médio')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(3,10,2), error2, color='magenta', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Taxa de erro(com normalização) X Valores de K')
    plt.xlabel('Valores para k')
    plt.ylabel('Erro médio')
    plt.show()

    # PASSO 4: CLASSIFICAÇÃO USANDO KNN

    knn1 = KNeighborsClassifier(n_neighbors=5)
    knn1.fit(previsores, classe)
    res1 = knn1.predict(previsores)

    knn2 = KNeighborsClassifier(n_neighbors=5)
    knn2.fit(previsores_normalizado, classe)
    res2 = knn2.predict(previsores_normalizado)

    # PASSO 5: MATRIZ DE CONFUSÃO
    from sklearn.metrics import confusion_matrix
    CM_res1=confusion_matrix(classe, res1)
    CM_res2=confusion_matrix(classe, res2)
    print("Sem normalização: ")
    print(CM_res1)
    print("Com normalização: ")
    print(CM_res2)

if __name__=="__main__":
    execute()

