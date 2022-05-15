import math
import operator
import random
from builtins import print

import numpy as np
# import pandas as pd
from random import randint
from itertools import permutations
from random import choices
import time as tim
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy

matrizcsv = np.genfromtxt("./csv/originales/deltas_5m.csv", delimiter=",")
movimientos = np.delete(matrizcsv, 0, 0)
movInicial = movimientos[0]
movimientos = np.delete(movimientos, 0, 0)
movimientos = movimientos * 2
# movimientos[0]=movimientos[0]/2
lista_mov = []
for fila in movimientos:
    for i, valor in enumerate(fila):
        # print(i,valor)
        if valor != 0:
            lista_mov.append([i, valor])

rows, col = movimientos.shape

matrizProximaIndices = np.genfromtxt("csv/originales/cercanas_indices.csv", delimiter=",")
matrizProximaIndices = np.delete(matrizProximaIndices, 0, 0)

matrizProximaDistancias = np.genfromtxt("csv/originales/cercanas_kms.csv", delimiter=",")
matrizProximaDistancias = np.delete(matrizProximaDistancias, 0, 0)


def randomlist(m, n):
    arr = np.zeros(m)
    for i in range(n):
        valor = randint(0, n)
        aux = valor % (m)
        arr[aux] += 1
    return arr


def modificar_estacion(accion, estacion, capacidadEstaciones, huecosLibres, bicisEstacion):
    # global bicisEstacion
    # Si queremos meter bicis
    if accion > 0:
        if huecosLibres[estacion] >= accion:
            bicisEstacion[estacion] += accion
            huecosLibres[estacion] -= accion
            # print("esto provoca errror 1")
            accion = 0

            return 0
        elif huecosLibres[estacion] > 0:
            while huecosLibres[estacion] > 0:
                huecosLibres[estacion] -= 1
                bicisEstacion[estacion] += 1
                accion -= 1
            # bicisEstacion[estacion]+= huecosLibres[estacion]
            # accion = accion-huecosLibres[estacion]
            # # bicisEstacion[estacion]= capacidadEstaciones[estacion]
            # huecosLibres[estacion] = 0
            # print("esto provoca error 2")
            return accion


    # Si queremos sacar bicis
    else:
        # bicisEstacion = capacidadEstaciones[estacion] - huecosLibres[estacion]
        if bicisEstacion[estacion] >= (abs(accion)):
            bicisEstacion[estacion] += accion
            huecosLibres[estacion] += abs(accion)
            accion = 0

            return accion
        elif bicisEstacion[estacion] > 0:
            # accion = accion + bicisEstacion[estacion]
            # # bicisEstacion[estacion]= 0
            # bicisEstacion[estacion]=0
            # # huecosLibres[estacion]+=num
            # huecosLibres[estacion] = capacidadEstaciones[estacion]
            # print("revienta")
            while bicisEstacion[estacion] > 0:
                bicisEstacion[estacion] -= 1
                huecosLibres[estacion] += 1
                accion += 1
            return accion

    return accion


def proxima(estacion, numero):
    return int(matrizProximaIndices[estacion, numero])


def ejecutar_accion(accion, estacion, capacidadEstaciones, huecosLibres, bicisEstacion):
    km = 0
    # global bicisEstacion
    if accion > 0:
        if huecosLibres[estacion] >= accion:
            # bicisEstacion[estacion]+=accion
            # huecosLibres[estacion]-=accion
            modificar_estacion(accion, estacion, capacidadEstaciones, huecosLibres, bicisEstacion)
            accion = 0

        elif huecosLibres[estacion] > 0:
            accion = modificar_estacion(accion, estacion, capacidadEstaciones, huecosLibres, bicisEstacion)
            # accion = accion - huecosLibres[estacion]
        i = 1
        while accion > 0:

            # print("estacion ",estacion," indice ",i)

            aux = modificar_estacion(accion, proxima(estacion, i), capacidadEstaciones, huecosLibres, bicisEstacion)
            # print("aux", aux)
            if aux < accion:
                bicis = accion - aux
                km += matrizProximaDistancias[estacion, i] * bicis
                accion = aux

            i += 1
    else:
        # bicisEstacion = capacidadEstaciones[estacion] - huecosLibres[estacion]
        if bicisEstacion[estacion] >= abs(accion):
            modificar_estacion(accion, estacion, capacidadEstaciones, huecosLibres, bicisEstacion)
            accion = 0
        elif bicisEstacion[estacion] > 0:
            accion = modificar_estacion(accion, estacion, capacidadEstaciones, huecosLibres, bicisEstacion)
            # accion=accion+bicisEstacion[estacion]
        j = 1
        while accion < 0:
            aux = modificar_estacion(accion, proxima(estacion, j), capacidadEstaciones, huecosLibres, bicisEstacion)
            # if aux is None:
            #     aux=0
            if aux > accion:
                bicis = abs(accion) - abs(aux)
                km += (matrizProximaDistancias[estacion, j]) * 3 * bicis
                accion = aux
            j += 1
    return km


def coste2(solucion_actual):
    capacidadEstaciones = solucion_actual.copy()
    huecosLibres = capacidadEstaciones.copy()
    bicisEstacion = np.array(capacidadEstaciones - huecosLibres)
    kmaux = 0
    for x in range(0, 16):
        ejecutar_accion(movInicial[x], x, capacidadEstaciones, huecosLibres, bicisEstacion)

    for elem in lista_mov:
        kmaux += ejecutar_accion(elem[1], elem[0], capacidadEstaciones, huecosLibres, bicisEstacion)

    return kmaux


def coste(solucion_actual, alpha):
    capacidadEstaciones = solucion_actual.copy()
    huecosLibres = capacidadEstaciones.copy()
    bicisEstacion = np.array(capacidadEstaciones - huecosLibres)
    kmaux = 0
    for x in range(0, 16):
        ejecutar_accion(movInicial[x], x, capacidadEstaciones, huecosLibres, bicisEstacion)

    for elem in lista_mov:
        kmaux += ejecutar_accion(elem[1], elem[0], capacidadEstaciones, huecosLibres, bicisEstacion)
    kmaux = kmaux + (np.array(capacidadEstaciones).sum() - 205) * alpha
    return kmaux


def genera_vecino_local(solucion_inicial, limite_vecinos, n, offset):
    solucion_actual = solucion_inicial.copy()
    permutaciones = list(permutations(np.arange(0, solucion_actual.shape[0]), 2))
    i = 0
    vecinos = []
    num_vecinos = 0
    long = len(permutaciones)
    # offset=randint(0,len(permutaciones))
    for x in range(offset, len(permutaciones) + offset):
        y = x % long
        if num_vecinos < limite_vecinos:
            if solucion_actual[permutaciones[y][1]] > n:
                sol_aux = solucion_actual.copy()
                sol_aux[permutaciones[y][0]] += n
                sol_aux[permutaciones[y][1]] -= n
                num_vecinos += 1
                vecinos.append(sol_aux.copy())
        else:
            break
    return vecinos


def mejor_coste(vecinos):
    minimo = math.inf
    for v in vecinos:
        if coste(v) < minimo:
            minimo = coste(v)
            vecino = v
    return vecino


def busqueda_local(solucion):
    inicio3 = tim.time()
    # genero la solucion inicial
    solucion_inicial = solucion
    # asigno la solucion inicial a la actual
    solucion_actual = solucion_inicial.copy()
    # asigno la solucion inicial a la solucion mejor
    mejor_solucion = solucion_actual.copy()
    i = 0
    mejora = True
    contador_mejora = 0
    offset = offset = randint(0, 240)
    tam_lote = 20
    num_vecinos = 240
    contador_costes = 0
    # print("la solucion inicial es: ", solucion_inicial)
    while i < 3000 and mejora:

        solucion_actual = mejor_coste(genera_vecino_local(solucion_actual, tam_lote, 2, offset))
        offset += tam_lote

        num_vecinos -= tam_lote
        contador_costes += 2
        if (coste(solucion_actual) < coste(mejor_solucion)):
            mejor_solucion = solucion_actual.copy()
            contador_mejora += 1
            num_vecinos = 240
        if num_vecinos == 0:
            mejora = False
        i += 1

    # print("contador mejoras local:", contador_mejora)
    # print("contador costes ", contador_costes)
    # print(mejor_solucion)
    # print("el tiempo de ", tim.time() - inicio3)
    return mejor_solucion


def genera_nueva_poblacion(elite, alpha):
    nueva_poblacion = elite.copy()
    for x in range(0, 25):
        aleatorio = randomlist(16, 220)

        nueva_poblacion.append(list([coste(aleatorio, alpha), aleatorio]))
    return nueva_poblacion


def genera_poblacion_inicial(alpha):
    nueva_poblacion = []
    for x in range(0, 30):
        aleatorio = randomlist(16, 220)

        nueva_poblacion.append(list([coste(aleatorio, alpha), aleatorio]))
    # nueva_poblacion.sort()
    nueva_poblacion = sorted(nueva_poblacion, key=lambda x: (x[0]))

    return nueva_poblacion


def distancia_hamming(padre, madre):
    distancia = 0
    for p, m in zip(padre, madre):
        if p != m:
            distancia += 1
    return distancia


def iguales(poblacion, nueva_poblaicon):
    iguales = True
    for p, np in zip(poblacion, nueva_poblaicon):
        if p[0] != np[0]:
            iguales = False
    return iguales


def mutacion_gaussiana(slot):
    modificacion = np.ceil(random.gauss(0, 2))
    if modificacion == 0:
        modificacion = 1
    if (slot + modificacion < 0):
        modificacion = abs(modificacion)
    slot += modificacion
    return slot


# print(mutacion_gaussiana(5))
def cruce(padre, madre):
    numeros = np.arange(0, 15)
    posiciones = choices(numeros, k=6)
    hijo = padre.copy()
    hija = madre.copy()
    for pos in posiciones:
        hija[pos] = padre[pos]
        hijo[pos] = madre[pos]
    return hijo, hija


def mutacion(hijo):
    posicion=randint(0, 15)
    slot=hijo[posicion]
    modificacion=np.ceil(random.gauss(0,2))
    if modificacion==0:
        modificacion=1
    if(slot+modificacion<0):
        modificacion=abs(modificacion)
    hijo[posicion] += modificacion

    # hijo[randint(0,15)]=randint(5,35)

    return hijo


def clearing(poblacion,radio,kappa):
    poblacion = sorted(poblacion, key=lambda x: (x[0]))

    Numganadores = 0
    indices = []
    for x in range(0, len(poblacion)):
        if poblacion[x][0] != 10000:
            Numganadores = 1
            for j in range(x + 1, len(poblacion)):
                # print(poblacion[i][1])

                dist = distancia_hamming(poblacion[x][1], poblacion[j][1])
                coste = poblacion[j][0]

                if coste > 0 and dist < radio:
                    if Numganadores < kappa:
                        Numganadores += 1
                    else:
                        poblacion[j][0] = 10000
                        indices.append(j)

    nueva_poblacion = []
    for p in poblacion:
        if p[0] != 10000:
            nueva_poblacion.append(list([p[0], p[1]]))

    return nueva_poblacion


def genetico_multimodal(tiempo, alpha,radio,kappa):
    random.seed(8721947)
    np.random.seed(8721947)

    poblacion = genera_poblacion_inicial(alpha)

    elite = poblacion[0:4]
    num_evaluaciones=30
    inicio = tim.time()
    iteracion = 0
    while (tim.time() - inicio) < tiempo:
        print(tim.time() - inicio)
        numeros = np.arange(0, 20)
        ultima = len(poblacion) - 1
        for i in range(0, 10):
            pesos = np.arange(len(poblacion) * 2, 0, -2)

            posiciones = choices(np.arange(0, len(poblacion)), weights=pesos, k=2)
            if random.uniform(0.0,1.0)>=0.8:
                hijo, hija = cruce(poblacion[posiciones[0]][1], poblacion[posiciones[1]][1])
                hijo = mutacion(hijo)
                hija = mutacion(hija)
                num_evaluaciones+=2
            else:
                hijo=poblacion[posiciones[0]][1]
                hija=poblacion[posiciones[1]][1]
            if len(poblacion) >= 30:
                if (np.array(hijo).sum() < 205):
                    poblacion[i + 5][0] = 1000
                else:
                    poblacion[i + 5][0] = coste(hijo, alpha)
                poblacion[i + 5][1] = hijo
                if (np.array(hija).sum() < 205):
                    poblacion[ultima - i][0] = 1000
                else:
                    poblacion[ultima - i][0] = coste(hija, alpha)
                poblacion[ultima - i][1] = hija

            else:
                if (np.array(hijo).sum() < 205):
                    poblacion[i + 5][0] = 1000
                    poblacion.append(list([10000, hijo]))
                else:
                    poblacion.append(list([coste(hijo, alpha), hijo]))
                if (np.array(hija).sum() < 205):
                    poblacion.append(list([10000, hija]))
                else:
                    poblacion.append(list([coste(hija, alpha), hija]))


        poblacion = sorted(poblacion, key=lambda x: (x[0]))

        elite = poblacion[0:5]

        if iteracion % 5 == 0:
            poblacion = clearing(poblacion,radio,kappa)
            poblacion = elite + poblacion
            poblacion = sorted(poblacion, key=lambda x: (x[0]))
            poblacion = poblacion[0:ultima]
        iteracion += 1

        plt.axis([0, tiempo, 0, 500])

        y = (poblacion[0][0] - ((poblacion[0][1].sum() - 205) * alpha))
        y2 = poblacion[0][1].sum()
        plt.plot(tim.time() - inicio, y, 'co')
        plt.plot(tim.time() - inicio, y2, 'ro')
        plt.legend(['km', 'slots'])
        plt.title('Multimodal')
        # plt.plot(tim.time() - inicio, y2,'ro')

        plt.pause(1)

    plt.show()

    km = poblacion[0][0] - (poblacion[0][1].sum() - 205) * alpha
    resultado = "fitness: " + str(poblacion[0][0]) + "  km: " + str(km) + "  slots: " + str(
        poblacion[0][1].sum()) + "  solucion: " + str(poblacion[0][1])
    print("numero de evaluaciones: ",num_evaluaciones)
    return resultado


print(genetico_multimodal(40, 4.5,4,2))
