import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

breakline = "############################################################"

data = pd.read_csv("./Data.csv", sep=",")
# print(data)


## Tratamento de dados

# Retirando valores negativos

condicao = data["duracao"] < 0
linhas = list(data[condicao].index.values)
data.drop(linhas, axis=0, inplace=True)

# verificando os clientes da base
arrayClientes = data.id_cliente.unique()

# Separando a base de dados por cliente

usuarios = []
for i in arrayClientes:
    condicao = data["id_cliente"] == i;
    cliente = data[condicao]
    usuarios.append(cliente)

## Criando variaveis para salvar os resultados
ProbGostar = []
MatrizesDeConfusao = []
Acuracia = []

for usuario in usuarios:
    # Retirando a coluna de id_cliente da base
    usuarioSemID = usuario.iloc[:, usuario.columns != "id_cliente"]

    # transformando as colunas de categorias em Dummy

    data_dummy = pd.get_dummies(usuarioSemID)

    ## separando o "like/deslike" dos dados em variaveis diferentes
    coluna = data_dummy.columns.get_loc("gostou")

    X = data_dummy.iloc[:, data_dummy.columns != "gostou"]
    Y = data_dummy.iloc[:, coluna]

    ## Separando os dados em dados de treino e de teste

    X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=1 / 5, random_state=0)

    ## Treinando o modelo

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    ## Previsão
    print(breakline)
    print()
    print(breakline)
    print()
    print("Usuario " + usuario.id_cliente.unique())

    y_pred_prob = classifier.predict_proba(X_test)

    probabilidade_de_gostar = y_pred_prob[:, 1]
    print(breakline)
    print("Probabilidade de gostar de determinada musica: ")
    print(probabilidade_de_gostar)
    ProbGostar.append(probabilidade_de_gostar)

    ########################
    ## MATRIZ DE CONFUSÃO ##
    ########################

    y_prev = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_prev)
    print(breakline)
    print("Matriz de confusão")
    print(cm)
    MatrizesDeConfusao.append(cm)

    #########################
    ## Acuracia dos testes ##
    #########################
    print(breakline)
    print("Precisão do modelo: ")
    print(accuracy_score(y_test, y_prev))
    Acuracia.append(accuracy_score(y_test, y_prev))

precisaoTotal = 0
for precisao in Acuracia:
    precisaoTotal += precisao
mediaDePrecisao = precisaoTotal / len(Acuracia)
print(mediaDePrecisao)




