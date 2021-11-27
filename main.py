import pandas as pd;
import sklearn.model_selection as ms
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

breakline = "############################################################"


data = pd.read_csv("./Data.csv", sep=",")
print(data)


## Tratamento de dados

# Retirando valores negativos

condicao = data["duracao"] < 0
linhas = list(data[condicao].index.values)
data.drop(linhas, axis=0, inplace = True)

# Removendo o id dos clientes da base
data.drop(["id_cliente"], axis='columns', inplace=True)


# transformando as colunas de categorias em Dummy

data_dummy = pd.get_dummies(data)

## separando o "like/deslike" dos dados em variaveis diferentes
coluna = data_dummy.columns.get_loc("gostou")

X = data_dummy.iloc[:,data_dummy.columns!="gostou"]
Y = data_dummy.iloc[:, coluna]



## Separando os dados em dados de treino e de teste

X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size = 1/5, random_state = 0)

## Treinando o modelo

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

## Previsão


y_pred_prob = classifier.predict_proba(X_test)

probabilidade_de_gostar = y_pred_prob[:, 1]
print(breakline)
print("Probabilidade de gostar de determinada musica: ")
print(probabilidade_de_gostar)



###########################
## MATRIZ DE CONFUSÃO ##
###########################

y_prev = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_prev)
print(breakline)
print("Matriz de confusão")
print(cm)


#########################
## Acuracia dos testes ##
#########################
print(breakline)
print("Precisão do modelo: ")
print(accuracy_score(y_test, y_prev))





