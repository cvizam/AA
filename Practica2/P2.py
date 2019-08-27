# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Christian Vigil Zamroa
"""
import numpy as np
import matplotlib.pyplot as plt
import random


# Fijamos la semilla
np.random.seed(1)

###############################################################################
# FUNCIONES EJERCICIO 1

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

###############################################################################
# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
    
print('1. EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO\n')
print('-> EJERCICIO 1\n')
print('* APARTADO A\n')

#CODIGO DEL ESTUDIANTE
unif = simula_unif(50, 2, [-50,50])
# Dibujo la gráfica con la nube de puntos
plt.scatter(unif[:,0],unif[:,1],c='g')
plt.title('Usando simula_unif')
plt.show()

print('\n* APARTADO B\n')

#CODIGO DEL ESTUDIANTE
gaus = simula_gaus(50, 2, np.array([5,7]))
# Dibujo la gráfica con la nube de puntos
plt.scatter(gaus[:,0],gaus[:,1],c='r')
plt.title('Usando simula_gaus')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente
print('-> EJERCICIO 2\n')

# FUNCIONES EJERCICIO 2

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

# Divide los datos en positivos o negativos
def divide_data(x,y):
    c1 = []
    c2 = []
    
    for i in range(len(x)):
        if y[i] == 1:
            c1.append(x[i])
        elif y[i] == -1:
            c2.append(x[i])
    return np.array(c1),np.array(c2)

# Divide los datos según su etiqueta
def divide_label(y):
    y1 = []
    y2 = []

    for i in range(len(y)):
        if y[i] == 1:
            y1.append(y[i])
        elif y[i] == -1:
            y2.append(y[i])
    return np.array(y1),np.array(y2)

# Introduce ruido sobre las etiquetas mediante aleatoriedad
def introduce_noise(labels):
    n = int(round((0.1*len(labels)))) # Para que sea al 10% de las etiquetas
    # Genero un vector con valores de 0 hasta el numero de etiquetas
    a = np.arange(len(labels))
    for i in range(n):
        # Selecciono un índice aleatorio al que acceder
        idx = random.choice(a)
        # Cambio el valor del elemento en ese índice
        if labels[idx] == 1:
            labels[idx] = -1
        else:
            labels[idx] = 1
    return np.array(labels,np.float64)

###############################################################################
#CODIGO DEL ESTUDIANTE
# APARTADO A)
print('* APARTADO A\n')

# Genero la muestra de puntos
data_a = simula_unif(50, 2, [-50,50])
# Inicializo las etiquetas a 0
label_a = np.zeros(len(data_a))
# Obtengo los términos de la recta
a,b = simula_recta(np.array([-50,50]))

# Añado las etiquetas usando el signo de la función
for i in range(len(data_a)):
    label_a[i] = f(data_a[i,0],data_a[i,1],a,b)
    
# Divido los datos en positivos y negativos
positivos,negativos = divide_data(data_a,label_a)

# Dibujo una gráfica de los puntos separados por su etiqueta y la recta usada.
plt.plot([-50,50],[a*-50+b,a*50+b],'k')
plt.scatter(positivos[:,0],positivos[:,1],c='g',label='Positivos')
plt.scatter(negativos[:,0],negativos[:,1],c='r',label='Negativos')
plt.title('Puntos y sus etiquetas')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE
# APARTADO B)
print('* APARTADO B\n')

# Divido las etiquetas en positivas y negativas
y1,y2 = divide_label(label_a)

# Introduzco ruido en un 10% de las etiquetas positivas y negativas
y1 = introduce_noise(y1)
y2 = introduce_noise(y2)

# Concateno los datos para separarlos de nuevo, tras la modificación de las etiquetas
data_b = np.concatenate((positivos,negativos),axis=0)
# Concateno las etiquetas una vez modificadas
label_b = np.concatenate((y1,y2),axis=0)

# Divido los datos de nuevo en positivos y negativos
positivos_b,negativos_b = divide_data(data_b,label_b)

# Dibujo una gráfica de los puntos separados por su etiqueta y la recta usada.
plt.plot([-50,50],[a*-50+b,a*50+b],'k')
plt.scatter(positivos_b[:,0],positivos_b[:,1],c='g',label='Positivos')
plt.scatter(negativos_b[:,0],negativos_b[:,1],c='r',label='Negativos')
plt.title('Puntos y sus etiquetas con ruido')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta
print('-> EJERCICIO 3\n')

# FUNCIONES EJERCICIO 3

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
  
# Almacena las 4 funciones y devuelve el resultado de evaluar unos puntos en la
# función que se indice mediante la variable 'i'
    
def functions(X,Y,i):
    functions = []
    functions.append((X-10)**2 + (Y-20)**2 - 400)
    functions.append(0.5*(X + 10)**2 + (Y - 20)**2 - 400)
    functions.append(0.5*(X - 10)**2 - (Y + 20)**2 - 400)
    functions.append(Y - 20*X**2 - 5*X + 3)
    
    return functions[i]

    
#CODIGO DEL ESTUDIANTE

i = 0
# Bucle para recorrer las 4 funciones
for i in range(4):
    
    # Genero una muestra de datos 
    x = np.linspace(-50.0,50.0,50)
    y = np.linspace(-50.0,50.0,50)
    # Convierto los arrays en matrices
    X,Y = np.meshgrid(x,y)
    
    # Obtengo los valores sobre los que se dibuja el contorno, evaluando 
    # los datos en la función dada por el índice 'i'
    Z = functions(X,Y,i)
    
    # Dibujo la gráfica
    plt.scatter(positivos_b[:,0],positivos_b[:,1],c='g', label='Positivos')
    plt.scatter(negativos_b[:,0],negativos_b[:,1],c='r', label='Negativos')
    # Dibujo el contorno
    plt.contour(X,Y,Z,[0],colors='black')
    plt.legend(loc='best')
    plt.title('Función {}'.format(i))
    plt.show()
    i += 1


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

print('2. MODELOS LINEALES\n')
# EJERCICIO 2.1: ALGORITMO PERCEPTRON

print('-> EJERCICIO 1\n')

# FUNCIONES EJERCICIO 2.1

# Añade una columna de 1's
def add_column(datos):
    i = 0
    caract = []
    for i in range(len(datos)):
        # Añado la columna de unos, y mantengo las 2 que ya estaban
        caract.append(np.array([1, datos[i][0], datos[i][1]]))
    return np.array(caract,np.float64)

# Algoritmo Perceptron
def ajusta_PLA(datos, label, max_iter, vini):
    # Almaceno una copia del valor inicial del vector de pesos
    w_ini = np.copy(vini)
    
    it = 0
    
    # Condición de salida por número de iteraciones
    while it < max_iter:
        # Incremento la iteración actual
        it += 1
        i = 0
        # Recorro los datos
        for i in range(len(datos)):
            # Perceptron. Si el signo del Perceptron no coincide con la
            # etiqueta del dato, se actualiza W.
            if signo(np.dot(np.transpose(vini),datos[i])) != label[i]:
                vini = vini + label[i] * datos[i]
        
        # Si no se ha producido ningún cambio en dicha iteración, se finaliza
        if np.array_equal(w_ini,vini):
            break;
        
        # Se guarda el vector actual antes de ser actualizado de nuevo.
        w_ini = np.copy(vini)
        
    return it, vini



#CODIGO DEL ESTUDIANTE
print('* APARTADO A\n')

# Variables para almacenar el numero medio de iteraciones
it_a = 0
it_b = 0

# Añado una columna de 1's a los datos
data_a = add_column(data_a)
# Repito el proceso 10 veces
for i in range(0,10):
    # Ejecución de PLA con vector cero
    it, w = ajusta_PLA(data_a,label_a,500,np.array([0.0,0.0,0.0]))
    # Almaceno las iteraciones
    it_a += it
    # Ejecución de PLA con vectores de números aleatorios
    it2, w = ajusta_PLA(data_a,label_a,500,np.random.rand(3))
    # Almaceno las iteraciones
    it_b += it2
    

# Muestro el número medio de iteraciones necesarias
print('Valor medio de iteraciones necesario para converger con vector cero: ',it_a/10)
print('Valor medio de iteraciones necesario para converger con vector aleatorio: ',it_b/10)

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE
print('* APARTADO B\n')

# Variables para almacenar el numero medio de iteraciones
it_a = 0
it_b = 0

# Añado una columna de 1's a los datos
data_b = add_column(data_b)
# Repito el proceso 10 veces
for i in range(0,10):
    # Ejecución de PLA con vector cero
    it, w = ajusta_PLA(data_b,label_b,500,np.array([0.0,0.0,0.0]))
    # Almaceno las iteraciones
    it_a += it
    # Ejecución de PLA con vectores de números aleatorios
    it2, w = ajusta_PLA(data_b,label_b,500,np.random.rand(3))
    # Almaceno las iteraciones
    it_b += it2
    

# Muestro el número medio de iteraciones necesarias    
print('Valor medio de iteraciones necesario para converger con vector cero: ',it_a/10)
print('Valor medio de iteraciones necesario para converger con vector aleatorio: ',it_b/10)


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
print('-> EJERCICIO 2\n')

# FUNCIONES PARA EL EJERCICIO 2

# Calculo del Gradiente
def gradient(x,y,w):
    return (-y * x) / (1 + np.exp(np.dot((y * x), w)))

# Función para estimar el error Eout.
def error(x,y,w):
    i = 0
    value = 0
    for i in range(len(x)):
        value += np.log(1 + np.exp(-y[i] * np.dot(np.transpose(w), x[i])))
    return value / len(x)

# Algoritmo de Regresión logística
def sgdRL(x,y,eta,maxIter):
    M = 64 # Tamaño de los minibatch
    # Inicializo el vector de pesos a 0
    w = np.array([0.0,0.0,0.0])
    # Guardo una copia del vector de pesos antes de comenzar las iteraciones
    w_old = np.copy(w)
    it = 0
    # Condición de salida por número de iteraciones
    while it < maxIter:
        # Reinicio el valor del Gradiente a 0
        value = np.array([0.0, 0.0, 0.0]) 
        # Permutación aleatoria de los datos
        cx = np.copy(x)
        cy = np.copy(y)
        perm = list(zip(cx,cy))
        np.random.shuffle(perm)
        nx, ny = zip(*perm)
        # Recorro el minibatch   
        for i in range(M):
            # Elemento aleatorio de la muestra
            n = np.random.randint(0,len(nx))
            # Calculo del Gradiente
            value += gradient(nx[n],ny[n],w)
            
        # Actualizo el vector de pesos
        w = w - eta * value
        
        # Compruebo si la norma vectorial entre el vector de pesos actual
        # y el vector de pesos anterios es menor que 0.01. Si lo es, finaliza         
        if np.linalg.norm(w_old - w) < 0.01:
            return w,it
        
        # Guardo el vector de pesos antes de pasar a la siguiente época
        w_old = np.copy(w)
        it += 1

    return w, it


#CODIGO DEL ESTUDIANTE
print('* APARTADO A\n')

# Genero los datos 
datos_2 = simula_unif(100, 2, [0,2])
# Inicializo el vector de etiquetas a 0
etiquetas_2 = np.zeros(len(datos_2))
# Obtengo los valores de la recta
a,b = simula_recta(np.array([0,2]))

# Añado las etiquetas usando el signo de la función
for i in range(len(datos_2)):
    etiquetas_2[i] = f(datos_2[i,0],datos_2[i,1],a,b)

# Divido los datos en positivos y negativos
pos,neg = divide_data(datos_2,etiquetas_2)

# Añado una columna de 1`s a los datos
datos_2 = add_column(datos_2)

# Ejecuto el algoritmo de Regresión Logística
w, it = sgdRL(datos_2,etiquetas_2,0.01,1000)

# Represento los datos y la recta obtenida
max_value = np.amax(datos_2) # Valor máximo de la muestra
t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t,'k-', label='recta reg. logística')
plt.plot([0,2],[a*0+b,a*2+b],'b-', label='simula_recta')
plt.scatter(pos[:,0],pos[:,1],c='g',label='Positivos')
plt.scatter(neg[:,0],neg[:,1],c='r',label='Negativos')
plt.ylim([-0.1,2.1])
plt.legend(loc='best')
plt.show()

print('\nError estimado para 100 muestras: ', error(datos_2,etiquetas_2,w))

input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

#CODIGO DEL ESTUDIANTE
print('* APARTADO B\n')

# Genero un muestra superior a 999
datos_2b = simula_unif(2000, 2, [0,2])
# Inicializo el vector de etiquetas a cero
etiquetas_2b = np.zeros(len(datos_2b))

# Añado las etiquetas usando el signo de la función
for i in range(len(datos_2b)):
    etiquetas_2b[i] = f(datos_2b[i,0],datos_2b[i,1],a,b)
   
# Divido los datos en positivos y negativos
pos1,neg1 = divide_data(datos_2b,etiquetas_2b)

# Añado una columna de 1's a los datos
datos_2b = add_column(datos_2b)

# Represento los datos y la recta obtenida
max_value = np.amax(datos_2b) # Valor máximo de la muestra
t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t,'k-', label='recta reg.logística')
plt.plot([0,2],[a*0+b,a*2+b],'b-', label='simula_recta')
plt.scatter(pos1[:,0],pos1[:,1],c='g',label='Positivos')
plt.scatter(neg1[:,0],neg1[:,1],c='r',label='Negativos')
plt.ylim([-0.1,2.1])
plt.legend(loc='lower right')
plt.show()

print('\nError estimado para 2000 muestras: ', error(datos_2b,etiquetas_2b,w))


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos
print('BONUS: CLASIFICACIÓN DE DÍGITOS\n')

# FUNCIONES PARA EL EJERCICIO BONUS

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Pseudoinversa	
def pseudoinverse(x,y):
    return np.dot(np.linalg.pinv(x),y)

# Calcula la tasa de error obtenida
def Ein(x,y,w):
  error = 0
  i = 0
  for i in range(len(x)):
      value = signo(np.dot(np.transpose(w),x[i]))
      if value != y[i]:
          error += 1
  
  return error/len(x)

# Calcula la cota
def cota(Ein,x,tol):
    N = len(x)
    loga = np.log((4*(2*N)**3 + 1) / tol)
    return Ein + np.sqrt((8*loga)/N)

#POCKET ALGORITHM
def ajusta_Pocket(x,y,maxIter,vini):
    # Almaceno el vector de pesos inicial
    w_ini = np.copy(vini)
    # Inicializo las iteraciones
    ite = 0
    # Calculo el error inicial
    e_ant = Ein(x,y,vini)
    
    # Condición de salida: cicla hasta llegar al máximo de iteraciones
    while ite < maxIter:
        ite += 1
        i = 0
        # Recorro los datos
        for i in range(len(x)):
            # Perceptron. Si el signo del Perceptron no coincide con la
            # etiqueta del dato, se actualiza W(vini.
            if signo(np.dot(np.transpose(vini),x[i])) != y[i]:
                vini = vini + y[i] * x[i]
        
        # Tras actualizar W(vini), calculo el error actual
        e_actual = Ein(x,y,vini)
        
        # Compruebo si el error actual es menor que el anterior
        if e_actual < e_ant:
            w_ini = np.copy(vini)
            e_ant = e_actual
        
    return w_ini

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

print('* APARTADO A\n')

#LINEAR REGRESSION FOR CLASSIFICATION 

print('-> REGRESIÓN LINEAL\n')
#CODIGO DEL ESTUDIANTE

# Ejecución del algoritmo de Clasificación Linear
w = pseudoinverse(x,y)

# Representación en una gráfica de los datos Train y su clasificación
fig, ax = plt.subplots()
max_value = np.amax(x) # Valor máximo de la muestra
t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t,'k-',label='función estimada') # Recta de regresión
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.ylim([-7.1,-0.5])
plt.legend()
plt.show()

# Representación en una gráfica de los datos Test y su clasificación
fig, ax = plt.subplots()
max_value = np.amax(x_test) # Valor máximo de la muestra
t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
plt.plot(t,-w[0]/w[2] - w[1]/w[2]*t,'k-',label='función estimada') # Recta de regresión
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.ylim([-7.1,-0.5])
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('\n-> ALGORITMO POCKET\n')

# Ejecución del algoritmo Pocket tomando como 'w' inicial, la obtenida en el
# algoritmo de Pseudo-Inversa
w1 = ajusta_Pocket(x,y,500,w)

# Representación en una gráfica de los datos Train y su clasificación
fig, ax = plt.subplots()
max_value = np.amax(x) # Valor máximo de la muestra
t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
plt.plot(t,-w1[0]/w1[2] - w1[1]/w1[2]*t,'k-', label='función estimada') # Recta de regresión
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.ylim([-7.1,-0.5])
plt.legend()
plt.show()

# Representación en una gráfica de los datos Train y su clasificación
fig, ax = plt.subplots()
max_value = np.amax(x_test) # Valor máximo de la muestra
t = np.arange(0.,max_value+0.5,0.5) # Valores desde 0 hasta max+0.5
plt.plot(t,-w1[0]/w1[2] - w1[1]/w1[2]*t,'k-',label='función estimada') # Recta de regresión
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.ylim([-7.1,-0.5])
plt.legend()
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

#CODIGO DEL ESTUDIANTE
print('* APARTADO B\n')

# Calculo el Ein para la Pseudo-Inversa
e_train_pseudo = Ein(x,y,w)
print('\nError obtenido para el Train con Pseudo-Inversa: ',e_train_pseudo)

# Calculo el Etest para la Pseudo-Inversa
e_test_pseudo = Ein(x_test,y_test,w)
print('\nError obtenido para el Test con Pseudo-Inversa: ',e_test_pseudo)

# Calculo el Ein para el Pocket
e_train_pocket = Ein(x,y,w1)
print('\nError obtenido para el Train con Pocket: ',e_train_pocket)

# Calculo el Etest para el Pocket
e_test_pocket = Ein(x_test,y_test,w1)
print('\nError obtenido para el Test con Pocket: ',e_test_pocket)



input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR
print('* APARTADO C\n')
#CODIGO DEL ESTUDIANTE

# Calculo la cota basada en Ein para el Pocket
cota_train_pocket = cota(e_train_pocket,x,0.05)
print('\nCota obtenida para el Train con Pocket: ',cota_train_pocket)

# Calculo la cota basada en Etest para el Pocket
cota_test_pocket = cota(e_test_pocket,x_test,0.05)
print('\nCota obtenida para el Test con Pocket: ',cota_test_pocket)



