# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Christian Vigil Zamora
"""

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(1) # Semilla 1


print('1. EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')

# Funciones empleadas en el Ejercicio 1

# Funcion E
def E(u,v):
    return np.float64(((u**2*np.e**v - 2*v**2*np.e**-u)**2)) 

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**2*np.e**v - 2*v**2*np.e**-u) * (2*u*np.e**v + 2*v**2*np.e**-u)
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**2*np.e**v - 2*v**2*np.e**-u) * (u**2*np.e**v - 4*v*np.e**-u)

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# Funcion F
def F(x,y):
    return np.float64((x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)))

# Derivada parcial de F con respecto de x
def dFu(x,y):
    return 2*x + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)

# Derivada parcial de F con respecto de y
def dFv(x,y):
    return 4*y + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

# Gradiente de F
def gradF(x,y):
    return np.array([dFu(x,y), dFv(x,y)])

# Algoritmo de Gradiente Descendente
def gradient_descent(f,df,eta,maxIter,error2get,initial_point,consider_error):
    w = initial_point # Inicializo w con el valor del punto inicial
    # Para controlar las iteraciones
    iterations = 0
    # Valor de la función en el punto inicial
    value = f(w[0],w[1])
    # Lista con los valores obtenido hasta alcanzar el mínimo
    values = []
    while iterations < maxIter: # Condición de salida
        iterations += 1
        # Actualizo las coordenadas restandole el valor del gradiente
        w = w - eta * df(w[0],w[1])
        # Obtengo el valor en esas coordenadas
        value = f(w[0],w[1])
        values.append(value)
        if value < error2get and consider_error: # Condicion de salida
            break;
    # Devuelve las coordenadas del punto minimo, su valor y la lista con
    # los valores que se han ido obteniendo en las coordenadas
    return w, iterations, values
    
# Funcion para mostrar plot 3D
def display_figure(f,w,title,x_as,y_as,z_as):
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-7.5, 7.5, 50)
    y = np.linspace(-7.5, 7.5, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y) #E_w([X, Y])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap='jet')
    min_point = np.array([w[0],w[1]])
    min_point_ = min_point[:, np.newaxis]
    ax.plot(min_point_[0], min_point_[1], f(min_point_[0], min_point_[1]), 'r*', markersize=10)
    ax.set(title=title)
    ax.set_xlabel(x_as)
    ax.set_ylabel(y_as)
    ax.set_zlabel(z_as)
    plt.show()


##########################################################################################################

print('EJERCICIO 1.2\n')

# APARTADO A)
print('* APARTADO A: \n')
eta = 0.01 # Learning rate
maxIter = 100 # Maximo de iteraciones
error2get = 1e-14 # Epsilon
initial_point = np.array([1.0,1.0]) # Punto inicial

# Llamada al algoritmo de Gradiente Descendente
w, it, values = gradient_descent(E,gradE,eta,maxIter,error2get,initial_point,True)
# Muestro el resultado de su ejecución
display_figure(E,w,'Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente','u','v','E(u,v)')

input("\n--- Pulsar tecla para continuar ---\n")

# APARTADO B)
print('\n* APARTADO B: \n')
print ('\nNumero de iteraciones: ', it)

input("\n--- Pulsar tecla para continuar ---\n")

# APARTADO C)
print('\n* APARTADO C: \n')
print ('\nCoordenadas obtenidas: (',w[0],',',w[1],')')
print('\nValor minimo obtenido: ', values[it-1])

input("\n--- Pulsar tecla para continuar ---\n")
##########################################################################################################
print('EJERCICIO 1.3\n')

# APARTADO A)
print('\n* APARTADO A: \n')
# Ultimo argumento en las llamadas al algoritmo Gradiente Descendiente 
# puesto a False, ya que no se usa Epsilon en este ejercicio.

# Usando taza de aprendizaje = 0.01
eta = 0.01 
maxIter = 50 # Maximo de iteraciones
error2get = 1e-14 # Epsilon
initial_point = np.array([0.1,0.1]) # Punto inicial
# Llamada al algortimo Gradiente Descendente
w, it, values = gradient_descent(F,gradF,eta,maxIter,error2get,initial_point,False)
values = np.array(values) # Lista con el valor de la función con las iteraciones

# Grafico de como desciende el valor de la función con las iteraciones
x = range(1,it+1) # Eje x con las iteraciones
plt.title('Usando tasa de aprendizaje = 0.01')
plt.plot(x,values)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Usando taza de aprendizaje = 0.1
eta = 0.1
# Llamada al algoritmo Gradiente Descendente
# Los argumentos maxIter, error2get e initial point los reutilizo del anterior
w, it, values = gradient_descent(F,gradF,eta,maxIter,error2get,initial_point,False)
values = np.array(values) # Lista con el valor de la función con las iteraciones

# Grafico de como desciende el valor de la función con las iteraciones
x = range(1,it+1)
plt.title('Usando tasa de aprendizaje = 0.1')
plt.plot(x,values)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# APARTADO B)
print('\n* APARTADO B: \n')

# Creo 'C', un array en el que almaceno los valores de las
# variables (x,y) que voy a ir obteniendo. Y 'M', un array en el 
# que voy almacenando valor mínimo obtenido considerando los diferentes 
# puntos de inicio.

C = np.zeros((8,1)) # Para almacenar las coordenadas obtenidas
M = np.zeros((4,1)) # Para almacenar los valores minimos obtenidos

# PUNTO DE INICIO (0.1,0.1)

# Llamada al algoritmo Gradiente Descendente
w, it, values = gradient_descent(F,gradF,0.01,50,error2get,np.array([0.1,0.1]),False)
C[0] = w[0]
C[1] = w[1]
M[0] = values[it-1]

# PUNTO DE INICIO (1.0,1.0)

# Llamada al algoritmo Gradiente Descendente
w, it, values = gradient_descent(F,gradF,0.01,50,error2get,np.array([1.0,1.0]),False)
C[2] = w[0]
C[3] = w[1]
M[1] = values[it-1]

# PUNTO DE INICIO (-0.5,-0.5)

# Llamada al algoritmo Gradiente Descendente
w, it, values = gradient_descent(F,gradF,0.01,50,error2get,np.array([-0.5,-0.5]),False)
C[4] = w[0]
C[5] = w[1]
M[2] = values[it-1]

# PUNTO DE INICIO (-1,-1)

# Llamada al algoritmo Gradiente Descendente
w, it, values = gradient_descent(F,gradF,0.01,50,error2get,np.array([-1.0,-1.0]),False)
C[6] = w[0]
C[7] = w[1]
M[3] = values[it-1]

print('\n--- Tabla con los valores obtenidos: ---\n')
print('\nP.Inicial  CoordenadaX    CoordenadaY     Minimo')
print('\n[0.1,0.1]  ',C[0],     '',C[1],      '',M[0])
print('\n[1.0,1.0]  ',C[2],     '',C[3],      '',M[1])
print('\n[-0.5,-0.5]',C[4],     '',C[5],      '',M[2])
print('\n[-1.0,-1.0]',C[6],     '',C[7],      '',M[3])

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################

print('2. EJERCICIO SOBRE REGRESION LINEAL\n')


label5 = 1
label1 = -1

# Funciones empleadas en el Ejercicio 2

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

  
# Función para calcular la funcion h
def H(x,w):
    # Producto vectorial de la traspuesta de w por una fila de la muestra
    return w.T @ x

# Funcion para calcular el error
def Err(x,y,w):
    value = 0
    i = 0
    for i in range(len(x)):
        value += np.power((w.T @ x[i])-y[i],2)
    return value/len(x)


# Gradiente Descendente Estocástico
def sgd(x,y,eta,maxIter,error2get):
    M = 64 # Tamaño de los minibatch
    w = np.array([0.0,0.0,0.0])
    k = len(x[0]) # Numero de componentes 
    iter = 0
    # Condición de salida por número de iteraciones
    while iter < maxIter:
        # Reinicio el valor del Gradiente a 0
        value = np.array([0.0, 0.0, 0.0]) 
        # Condición de salida por Error menor que epsilon
        if(Err(x,y,w) < error2get):
                return w
        # Recorro el minibatch   
        for i in range(M):
            n = np.random.randint(0,len(x)) # Elemento aleatorio de la muestra
            # Recorro los componentes de x y del Gradiente
            for j in range(k): 
                # Calculo del Gradiente para ese minibatch
                value[j] += x[n,j] * (H(x[n],w)-y[n]) 
        # Actualizo la posición hacia el mínimo      
        w = w - eta * value  
        iter += 1
        
    return w

# Pseudoinversa	
def pseudoinverse(x,y):
    return np.dot(np.linalg.pinv(x),y)

# Pintar las soluciones obtenidas junto con los datos usados en el ajuste
def display_regression(x,a,b,w,title):
    max_value = np.amax(x) # Valor máximo de la muestra
    t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
    plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t,'k-') # Recta de regresión
    plt.scatter(a[:,1:2],a[:,2:3],c='r',label='5') # Datos de tipo 5
    plt.scatter(b[:,1:2],b[:,2:3],c='b',label='1') # Datos de tipo 1
    plt.title(title)
    plt.xlabel('Intensidad promedio')
    plt.ylabel('Simetria')
    plt.legend()
    plt.show()
    

##########################################################################################################
print('EJERCICIO 1\n')

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')
 
# División de los datos segun su clase, para poder representarlos
clase_1 = np.zeros((len(x),3))
clase_5 = np.zeros((len(x),3))
i1 = 0
i5 = 0
for c_1 in range(len(x)):
    if y[c_1] == 1:
        clase_5[i5] = x[c_1]
        i5 += 1
    elif y[c_1] == -1:
        clase_1[i1] = x[c_1]
        i1 += 1


print ('Bondad del resultado para pseudoinversa:\n')
# Uso el algoritmo de la pseudo-inversa
w = pseudoinverse(x,y)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
# Pinto la solución junto con los datos usados en el ajuste
display_regression(x,clase_5,clase_1,w,'Pseudo-inversa')

input("\n--- Pulsar tecla para continuar ---\n")

print ('Bondad del resultado para grad. descendente estocastico:\n')
# Uso el algoritmo de Gradiente Descendente Estocástico
# Learning rate = 0.001, maxIter = 100, error2get = 0.085
w = sgd(x,y,0.001,100,0.085) 
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
# Pinto la solución junto con los datos usados en el ajuste
display_regression(x,clase_5,clase_1,w,'Grad. descendente estocastico')


input("\n--- Pulsar tecla para continuar ---\n")
##########################################################################################################
print('EJERCICIO 2\n')

# Funciones usadas en el Ejercicio 2

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Asigna una etiqueta a cada punto de la muestra
def assign_label(train):
    label = np.zeros(train.shape[0])
    for i in range(len(train)):
        label[i] = np.sign(np.power((train[i][0:1]-0.2),2) + np.power(train[i][1:2],2) - 0.6)

    label = np.array(label,np.float64)
    return label

# Introduce ruido sobre las etiquetas mediante aleatoriedad
def introduce_noise(train,labels):
    n = int(0.1*len(train)) # Para que sea al 10% de las etiquetas
    # Genero un vector con valores de 0 hasta el numero de muestras
    a = np.arange(len(train))
    for i in range(n):
        # Selecciono un índice aleatorio al que acceder
        idx = random.choice(a)
        # Cambio el valor del elemento en ese índice
        if labels[idx] == 1:
            labels[idx] = -1
        else:
            labels[idx] = 1
    return np.array(labels,np.float64)

# Añade una columna de 1's
def add_column(train):
    i = 0
    caract = []
    for i in range(len(train)):
        # Añado la columna de unos, y mantengo las 2 que ya estaban
        caract.append(np.array([1, train[i][0], train[i][1]]))
    return np.array(caract,np.float64)

# Divide los datos según su etiqueta
def divide_per_label(train,labels):
    c1 = np.zeros((len(train),2))
    c2 = np.zeros((len(train),2))
    i1 = 0
    i2 = 0
    for i in range(len(train)):
        if labels[i] == 1:
            c1[i1] = train[i]
            i1 += 1
        elif labels[i] == -1:
            c2[i2] = train[i]
            i2 += 1
    return c1,c2
##########################################################################################################
# APARTADO a)
print('\n* APARTADO A: \n')

# Genero la muestra de entrenamiento 
train = simula_unif(1000,2,1)
# Pinto el mapa de puntos 2D
plt.scatter(train[:,0],train[:,1],c='r')
plt.title('Mapa de puntos 2D')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# APARTADO b)
print('\n* APARTADO B: \n')

# Asigno una etiqueta a cada punto de la muestra 'train'
# considerando la función dada. A continuación, introduzco
# ruido sobre las etiquetas mediante aleatoriedad. Por último,
# pinto el mapa de etiquetas junto a la muestra.

labels = assign_label(train)
labels = introduce_noise(train,labels)


# Divido los datos de la muestra según su etiqueta, para así
# poder representarlos correctamente.

c1,c2 = divide_per_label(train,labels)

# Pinto el mapa de etiquetas
plt.scatter(c1[:,0],c1[:,1],c='g',label='1')
plt.scatter(c2[:,0],c2[:,1],c='y',label='-1')
plt.title('Mapa de etiquetas')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

####################################
# APARTADO c)
print('\n* APARTADO C: \n')

caract = add_column(train)
print ('Bondad del resultado para grad. descendente estocastico:\n')
# Uso el algoritmo Gradiente Descendente Estocástico
# Learning rate = 0.001, iteraciones máximas = 100, epsilon = 0.095
w = sgd(caract,labels,0.001,100,0.095)
print ("Ein: ", Err(caract,labels,w))

# Divido los datos de la muestra según su etiqueta, para así
# poder representarlos correctamente.

c1,c2 = divide_per_label(train,labels)

# Pinto el mapa de etiquetas
max_value = np.amax(x) # Valor máximo de la muestra
t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t,'k-') # Recta de regresión
plt.scatter(c1[:,0],c1[:,1],c='g',label='1')
plt.scatter(c2[:,0],c2[:,1],c='y',label='-1')
plt.axis([-1.1,1.1,-1.1,1.1]) # Redimensiono ejes, para visualizar bien
plt.title('Ajuste modelo de Regresión Lineal')
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

####################################
# APARTADO d)

print('\n* APARTADO D: \n')
print('\n-- Tiempo de cálculo aproximado: 50-60s --\n')

# Variables para almacenar el Ein e Eout medio
Ein_medio = 0.0 
Eout_medio = 0.0

# Bucle para ejecutar el experimento 1000 veces
for i in range(1000):
    # Genero muestra de entrenamiento
    data = simula_unif(1000,2,1) 
    # Asigno una etiqueta a cada punto de la muestra
    etiq = assign_label(data) 
    # Introduzco ruido sobre el 10 % de las etiquetas
    etiq = introduce_noise(data,etiq)
    # Añado una columna de 1's a la muestra
    carac = add_column(data)
    # Uso el algoritmo Gradiente Descendente
    # Learning rate = 0.01, iteraciones máximas = 10, epsilon = 0.095
    w = sgd(carac,etiq,0.01,10,0.095)
    # Sumatoria del error Ein obtenido en cada iteración
    Ein_medio += Err(carac,etiq,w)
    # Genero 1000 puntos nuevos
    data_test = simula_unif(1000,2,1)
    # Asigno una etiqueta a cada uno de ellos
    etiq_test = assign_label(data_test)
    # Introduzco ruido sobre el 10 % de las etiquetas
    etiq_test = introduce_noise(data_test,etiq_test)
    # Añado una columna de 1's a la muestra
    carac_test = add_column(data_test)
    # Sumatoria del error Eout obtenido en cada iteración
    Eout_medio += Err(carac_test,etiq_test,w)

# Representación de los valores medios de los errores Ein e Eout obtenidos 
    
print("Valor medio Ein de las 1000 muestras: ", Ein_medio/1000)
print("Valor medio Ein de las 1000 muestras: ", Eout_medio/1000)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################

print('BONUS\n')
print('EJERCICIO 1\n')

# Derivadas necesarias para el calculo de la matriz Hessiana

def dFxx(x,y):
    return 2-8*np.pi**2*np.sin(2*np.pi*y)*np.sin(2*np.pi*x)

def dFyy(x,y):
    return 4-8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def dFxy(x,y):
    return 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

def dFyx(x,y):
    return 8*np.pi**2*np.cos(2*np.pi*y)*np.cos(2*np.pi*x)

# Método de Newton
def NM(f,gradF,eta,maxIter,initial_point,error2get):
    w = initial_point 
    # Lista con los valores obtenidos en busca del mínimo
    values = [] 
    it = 0
    
    # Condición de salida
    while it < maxIter:
        it += 1
        
        # Guardo el valor de evaluar w en la función, antes de actualizar w
        old = f(w[0],w[1])
        
        # Primer elemento de la matriz Hessiana formado por
        # la 2da derivada de x y la derivada de xy
        el1 = np.array([dFxx(w[0],w[1]),dFxy(w[0],w[1])])
        
        # Segundo elemento de la matriz Hessiana formado por
        # la derivada de yx y la 2da derivada de y
        el2 = np.array([dFyx(w[0],w[1]),dFyy(w[0],w[1])])
        
        # Matriz Hessiana
        H = np.array([el1,el2])
    
        # Actualizo la posición
        w = w - eta * (np.linalg.inv(H) @ gradF(w[0],w[1]))
        
        # Obtengo el valor de f en la nueva w
        value = f(w[0],w[1])
        values.append(value)
    
        # Condición de salida
        if old - value < error2get:
            break;
        
    return w, it, values

##########################################################################################################   
# APARTADO A)
print('\n* APARTADO A: \n')

# Usando taza de aprendizaje = 0.01
eta = 0.01
maxIter = 50 # Maximo de iteraciones
error2get = 1e-14 # Epsilon
initial_point = np.array([0.1,0.1]) # Punto inicial
# Llamada al Método de Newton
w, it, values = NM(F,gradF,eta,maxIter,initial_point,error2get)
values = np.array(values) # Lista con el valor de la función con las iteraciones

# Grafico de como desciende el valor de la función con las iteraciones
x = range(1,it+1) # Eje x con las iteraciones
plt.title('Usando tasa de aprendizaje = 0.01\n Método de Newton')
plt.plot(x,values)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Usando taza de aprendizaje = 0.1
eta = 0.1
# Llamada al Método de Newton
# Los argumentos maxIter, error2get e initial point los reutilizo del anterior
w, it, values = NM(F,gradF,eta,maxIter,initial_point,error2get)
values = np.array(values) # Lista con el valor de la función con las iteraciones

# Grafico de como desciende el valor de la función con las iteraciones
x = range(1,it+1)
plt.title('Usando tasa de aprendizaje = 0.1\n Método de Newton')
plt.plot(x,values)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# APARTADO B)
print('\n* APARTADO B: \n')

# Creo 'C2', un array en el que almaceno los valores de las
# variables (x,y) que voy a ir obteniendo. Y 'M2', un array en el 
# que voy almacenando valor mínimo obtenido considerando los diferentes 
# puntos de inicio.

C2 = np.zeros((8,1)) # Para almacenar las coordenadas obtenidas
M2 = np.zeros((4,1)) # Para almacenar los valores minimos obtenidos

# PUNTO DE INICIO (0.1,0.1)

# Llamada al algoritmo al Método de Newton
w, it, values = NM(F,gradF,0.1,50,np.array([0.1,0.1]),error2get)
C2[0] = w[0]
C2[1] = w[1]
M2[0] = values[it-1]

# PUNTO DE INICIO (1.0,1.0)

# Llamada al algoritmo al Método de Newton
w, it, values = NM(F,gradF,0.1,50,np.array([1.0,1.0]),error2get)
C2[2] = w[0]
C2[3] = w[1]
M2[1] = values[it-1]

# PUNTO DE INICIO (-0.5,-0.5)

# Llamada al algoritmo al Método de Newton
w, it, values = NM(F,gradF,0.1,50,np.array([-0.5,-0.5]),error2get)
C2[4] = w[0]
C2[5] = w[1]
M2[2] = values[it-1]

# PUNTO DE INICIO (-1,-1)

# Llamada al Método de Newton
w, it, values = NM(F,gradF,0.1,50,np.array([-1.0,-1.0]),error2get)
C2[6] = w[0]
C2[7] = w[1]
M2[3] = values[it-1]

print('\n--- Tabla con los valores obtenidos: ---\n')
print('\nP.Inicial  CoordenadaX    CoordenadaY     Minimo')
print('\n[0.1,0.1]  ',C2[0],     '',C2[1],      '',M2[0])
print('\n[1.0,1.0]  ',C2[2],     '',C2[3],      '',M2[1])
print('\n[-0.5,-0.5]',C2[4],     '',C2[5],      '',M2[2])
print('\n[-1.0,-1.0]',C2[6],     '',C2[7],      '',M2[3])








