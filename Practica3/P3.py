# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:38:53 2019

@author: Christian
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Semilla Utilizada

np.random.seed(1)

###############################################################################
# FUNCIONES EMPLEADAS

# Para cargar los datos
def load(rute):
    data = pd.read_csv(rute,header=-1)
    data = np.array(data)

    x = data[:,:-1]
    y = data[:,-1]
  
    return np.array(x) , np.array(y)

# Para visualizar 6 datos del conjunto
def display_data(x):
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(np.reshape(x[i], (8,8)), cmap='summer')
    plt.show()
    
# Para mostrar la distribución de los dígitos
def show_distribution(y):
    data_0 = np.count_nonzero(y==0)
    data_1 = np.count_nonzero(y==1)
    data_2 = np.count_nonzero(y==2)
    data_3 = np.count_nonzero(y==3)
    data_4 = np.count_nonzero(y==4)
    data_5 = np.count_nonzero(y==5)
    data_6 = np.count_nonzero(y==6)
    data_7 = np.count_nonzero(y==7)
    data_8 = np.count_nonzero(y==8)
    data_9 = np.count_nonzero(y==9)
    
    labels = ['0','1','2','3','4','5','6','7','8','9']
    data = [data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9]
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title('Frecuencia de aparición de dígitos')
    plt.show()

# Para mostrar una tabla con la distribución de los dígitos del Test
def show_test_distribution(y):
    data_0 = np.count_nonzero(y==0)
    data_1 = np.count_nonzero(y==1)
    data_2 = np.count_nonzero(y==2)
    data_3 = np.count_nonzero(y==3)
    data_4 = np.count_nonzero(y==4)
    data_5 = np.count_nonzero(y==5)
    data_6 = np.count_nonzero(y==6)
    data_7 = np.count_nonzero(y==7)
    data_8 = np.count_nonzero(y==8)
    data_9 = np.count_nonzero(y==9)
    
    print('Número de 0`s = ',data_0)
    print('Número de 1`s = ',data_1)
    print('Número de 2`s = ',data_2)
    print('Número de 3`s = ',data_3)
    print('Número de 4`s = ',data_4)
    print('Número de 5`s = ',data_5)
    print('Número de 6`s = ',data_6)
    print('Número de 7`s = ',data_7)
    print('Número de 8`s = ',data_8)
    print('Número de 9`s = ',data_9)
    
###############################################################################
# PROBLEMA DE CLASIFICACIÓN

print('\n1. PROBLEMA DE CLASIFICACIÓN')

# Lectura de los datos de Entrenamiento
x_train, y_train = load('datos/optdigits.tra')

# Lectura de los datos de Test
x_test, y_test = load('datos/optdigits.tes')

# VISUALIZACIÓN DE LOS CONJUNTOS DE ENTRENAMIENTO Y TEST

# Primero los datos del conjunto de Entrenamiento
print('\n-> Datos Train: ')
display_data(x_train)

# Correspondencia de las etiquetas con los datos mostrados
print('\n-> Etiquetas correspondientes: \n')
print(y_train[0:6])

input("\n--- Pulsar tecla para continuar ---\n")

# Compruebo si los datos están distribuidos equitativamente
print('\n-> Distribución de dígitos según su clase')
show_distribution(y_train)

input("\n--- Pulsar tecla para continuar ---\n")

# En segundo lugar, los datos del conjunto de Test
print('\n-> Datos Test: ')
display_data(x_test)

# Correspondencia de las etiquetas con los datos mostrados
print('\n-> Etiquetas correspondientes: \n')
print(y_test[0:6])

input("\n--- Pulsar tecla para continuar ---\n")

# Compruebo si los datos están distribuidos equitativamente
print('\n-> Distribución de dígitos según su clase')
show_distribution(y_test)


# PREPROCESADO DE LOS DATOS

# Eliminación de datos sin variabilidad

selector = VarianceThreshold(threshold=0.1)

# Ajuste y transformación de los datos de Entrenamiento
x_train = selector.fit_transform(x_train)

# Ajuste y transformación de los datos de Test
x_test = selector.fit_transform(x_test)


# Normalización de los datos

x_train = x_train * 1.0
x_test = x_test * 1.0
x_train = (MinMaxScaler().fit_transform(x_train))
x_test = (MinMaxScaler().fit_transform(x_test))

# Estandarización de los datos

x_train = (StandardScaler().fit_transform(x_train))
x_test = (StandardScaler().fit_transform(x_test))

# DEFINICIÓN DE LOS MODELOS LINEALES A USAR Y ESTIMACIÓN DE SUS PARÁMETROS

input("\n--- Pulsar tecla para continuar ---\n")

print('\n-> Modelos Lineales a usar: ')

print('\n-> Regresión Logística')

# Parámetros a evaluar
parameters = [{'penalty':['l2'], 'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'tol':[1e-2,1e-3,1e-4,1e-5]}]
# Algoritmo usado
logistic = LogisticRegression(max_iter=500, random_state=1,solver='newton-cg', multi_class='auto')
# Obtención del modelo con los mejores parámetros
logistic_model = GridSearchCV(logistic, parameters, cv=5, scoring = 'accuracy')
logistic_model.fit(x_train,y_train)

# Parámetros elegidos y Rendimiento obtenido
print('\n* Mejores parámetros: ',logistic_model.best_params_)
print('\n* Mejor precisión: ',logistic_model.best_score_)
print('\n* Ein medio: ',1 - logistic_model.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

print('\n-> Perceptron')

# Parámetros a evaluar
parameters_pla = [{'penalty':['l1','l2'], 'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'tol':[1e-2,1e-3,1e-4,1e-5]}]
# Algoritmo usado
pla = Perceptron(random_state=1, max_iter=500)
# Obtención del modelo con los mejores parámetros
PLA_model = GridSearchCV(pla, parameters_pla, cv=5, iid = False, scoring = 'accuracy')
PLA_model.fit(x_train,y_train)

# Parámetros elegidos y Rendimiento obtenido
print('\n* Mejores parámetros: ',PLA_model.best_params_)
print('\n* Mejor precisión: ',PLA_model.best_score_)
print('\n* Ein medio: ',1 - PLA_model.best_score_)


input("\n--- Pulsar tecla para continuar ---\n")

# SELECCIÓN Y AJUSTE DEL MODELO FINAL

print('\n-> Modelo seleccionado: Regresión Logística')

# Predicción de las etiquetas usando el modelo eligido, sobre los datos de Test
prediction = logistic_model.predict(x_test)

# Resultado de la predicción en base a las etiquetas de los datos Test
score = logistic_model.score(x_test,y_test)

print('\n* Precisión obtenida sobre el conjunto Test: ',score)

input("\n--- Pulsar tecla para continuar ---\n")

# ESTIMACIÓN DEL ERROR EOUT

print('\n-> Eout obtenido: ',1 - score)

input("\n--- Pulsar tecla para continuar ---\n")

# JUSTIFICACIÓN DE LA CALIDAD DEL MODELO

print('\n-> Muestro la cantidad de dígitos de cada clase en el conjunto test\n')

# Cantidad de dígitos de cada clase en el conjunto de Test
show_test_distribution(y_test)

input("\n--- Pulsar tecla para continuar ---\n")

print('\n-> Compruebo la calidad del modelo mediante la Matriz de Confusión')

# Matriz de Confusión para comparar las etiquetas predichas con las etiquetas
# originales del conjunto de Test

cm = metrics.confusion_matrix(y_test, prediction)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'PuBu_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

###############################################################################
# PROBLEMA DE REGRESIÓN

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
# FUNCIONES

# Para visualizar los datos
def airfoil_data(data, atributtes):
    n = data.shape[1]
    figure, axes = plt.subplots(figsize=(9,5),nrows=n, ncols=n, sharex='col', sharey='row')
    for i in range(n):
        for j in range(n):
            axe = axes[i,j]
            if i == j:
                axe.text(0.5, 0.5, atributtes[i], transform = axe.transAxes, 
                         horizontalalignment = 'center', 
                         verticalalignment = 'center', fontsize = 9)
            else:
                axe.plot(data[:,j], data[:,i],'.k', color='k' )
    plt.show()

###############################################################################
print('\n2. PROBLEMA DE REGRESIÓN')

# Lectura de datos
data = np.loadtxt('datos/airfoil_self_noise.dat')

# Separo los datos en Características(x) y Etiquetas(y)

x_airfoil = data[:,:-1]
y_airfoil = data[:,-1]

# Las convierto en arrays de Numpy

x_airfoil = np.array(x_airfoil)
y_airfoil = np.array(y_airfoil)


# Defino los atributos de los datos para su visualización

atributtes = ['Frequency', 'Angle', 'Length', 'Velocity', 'Thickness', 'Sound']

# VISUALIZACIÓN LOS DATOS

print('\n-> Visualización de los datos')

airfoil_data(data, atributtes)

input("\n--- Pulsar tecla para continuar ---\n")

# PREPROCESADO DE LOS DATOS

# Transformación de Yeo-Johnson

x_airfoil = (PowerTransformer().fit_transform(x_airfoil))

# Añado caracteristicas incrementando la clase de funciones
# Mejor opción obtenida: polinomios de Grado 3

poly = PolynomialFeatures(3)
x_airfoil = poly.fit_transform(x_airfoil)

# DEFINICIÓN DE LOS CONJUNTOS DE TRAINING Y TEST

x_airfoil_train, x_airfoil_test, y_airfoil_train, y_airfoil_test = train_test_split(x_airfoil, y_airfoil, test_size = 0.2, random_state = 1)


# DEFINICIÓN DE LOS MODELOS LINEALES A USAR Y ESTIMACIÓN DE SUS PARÁMETROS
print('\n-> Modelos Lineales a usar: ')

print('\n-> Ridge')

# Parámetros a evaluar
parameters = [{'alpha':[0.001, 0.01, 0.1, 1, 10, 100], 'tol':[1e-2,1e-3,1e-4,1e-5]}]
# Algoritmo a usar
ridge = Ridge()
# Obtención del modelo con los mejores parámetros
RIDGE_model = GridSearchCV(ridge, parameters, cv=5, return_train_score=True, scoring = 'r2')
RIDGE_model.fit(x_airfoil_train,y_airfoil_train)

# Parámetros elegidos y Rendimiento obtenido
print('\n* Mejores parámetros: ',RIDGE_model.best_params_)
print('\n* Coeficiente de Determinación: ',RIDGE_model.best_score_)
print('\n* Ein medio: ',1 - RIDGE_model.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

print('\n-> Lasso')

# Parámetros a evaluar
parameters = [{'alpha':[0.001, 0.01, 0.1, 1, 10, 100], 'tol':[1e-2,1e-3,1e-4,1e-5]}]
# Algoritmo a usar
lasso = Lasso(max_iter=100000)
# Obtención del modelo con los mejores parámetros
LASSO_model = GridSearchCV(lasso, parameters, cv=5, return_train_score=True, scoring = 'r2')
LASSO_model.fit(x_airfoil_train,y_airfoil_train)

# Parámetros elegidos y Rendimiento obtenido
print('\n* Mejores parámetros: ',LASSO_model.best_params_)
print('\n* Coeficiente de Determinación: ',LASSO_model.best_score_)
print('\n* Ein medio: ',1 - LASSO_model.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

print('\n-> Linear Regression')

# Parámetros a evaluar
parameters = [{'normalize':['True','False','optional']}]
# Algoritmo a usar
lr = LinearRegression()
# Obtención del modelo con los mejores parámetros
LR_model = GridSearchCV(lr, parameters, cv=5, return_train_score=True, scoring = 'r2')
LR_model.fit(x_airfoil_train,y_airfoil_train)

# Parámetros elegidos y Rendimiento obtenido
print('\n* Mejores parámetros: ',LR_model.best_params_)
print('\n* Coeficiente de Determinación: ',LR_model.best_score_)
print('\n* Ein medio: ',1 - LR_model.best_score_)


input("\n--- Pulsar tecla para continuar ---\n")

# SELECCIÓN Y AJUSTE DEL MODELO FINAL

print('\n-> Modelo Seleccionado: Ridge')

# Predicción de las etiquetas usando el modelo eligido, sobre los datos de Test
prediction = RIDGE_model.predict(x_airfoil_test)

# Resultado de la predicción en base a las etiquetas de los datos Test
score = RIDGE_model.score(x_airfoil_test,y_airfoil_test)

print('\n* Coeficiente de Determinación obtenido sobre el conjunto Test: ',score)


# ESTIMACIÓN DEL ERROR EOUT

input("\n--- Pulsar tecla para continuar ---\n")

print('\n-> Eout obtenido: ',1 - score)

input("\n--- Pulsar tecla para continuar ---\n")

# JUSTIFICACIÓN DE LA CALIDAD DEL MODELO

print('\n-> Compruebo la calidad del modelo')
plt.scatter(y_airfoil_test, prediction, c='r')
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Model Quality')
plt.show()





