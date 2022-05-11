import math
import operator
import random
from builtins import print

import numpy as np
# import pandas as pd
from random import randint
from itertools import permutations
from random import choices
from scipy.stats import geom
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


# def coste(solucion_actual):
#     capacidadEstaciones = solucion_actual.copy()
#     huecosLibres = capacidadEstaciones.copy()
#     bicisEstacion = np.array(capacidadEstaciones - huecosLibres)
#     kmaux = 0
#     for x in range(0, 16):
#         ejecutar_accion(movInicial[x], x, capacidadEstaciones, huecosLibres, bicisEstacion)
#
#     for elem in lista_mov:
#         kmaux += ejecutar_accion(elem[1], elem[0], capacidadEstaciones, huecosLibres, bicisEstacion)
#
#     return kmaux


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


def genera_nueva_poblacion(elite,alpha):
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
    return nueva_poblacion

def distancia_hamming(padre,madre):
    distancia=0
    indices=[]
    for i,(p,m) in enumerate(zip(padre,madre)):
        if p!=m:
            distancia+=1
            indices.append(i)
    return indices,distancia
def iguales(poblacion,nueva_poblaicon):
    iguales=True
    for p,np in zip(poblacion,nueva_poblaicon):
        if p[0]!=np[0]:
            iguales=False
    return iguales

def mutacion_gaussiana(slot):
    modificacion=np.ceil(random.gauss(0,2))
    if modificacion==0:
        modificacion=1
    if(slot+modificacion<0):
        modificacion=abs(modificacion)
    slot+=modificacion
    return  slot
# print(mutacion_gaussiana(5))
def cruce_chc(padre,madre,indicies,distancia_hamming_padres):
    mutaciones_hijo=np.floor(distancia_hamming_padres/2)
    hijo=padre.copy()
    hija=madre.copy()
    x=0
    for i in indicies:

        if x % 2 ==0:
            hijo[i]=mutacion_gaussiana(madre[i])
        else:
            hija[i]=mutacion_gaussiana(padre[i])
        x+=1
    return hijo,hija
def chc(tiempo,alpha):
    poblacion=genera_poblacion_inicial(alpha)
    distancia_umbral=4
    inicio=tim.time()
    reinicios=0

    while tim.time()-inicio<tiempo:
        print(tim.time() - inicio)
        if(distancia_umbral==0):
            reinicios+=1
            poblacion=deepcopy(genera_nueva_poblacion(deepcopy(poblacion[0:5]),alpha))
            distancia_umbral=4
        random.shuffle(poblacion)
        nueva_poblacion=deepcopy(poblacion)
        for i in range(0,15):
            # print(len(poblacion))
            indices,distancia_hamming_padres=distancia_hamming(poblacion[i][1].tolist(), poblacion[29 - i][1].tolist())
            if(distancia_hamming_padres>distancia_umbral):
                hijo,hija=cruce_chc(poblacion[i][1],poblacion[29-i][1],indices,distancia_hamming_padres)
                if(hijo.sum()<205):
                    nueva_poblacion.append(list([1000,hijo]))
                else:
                    nueva_poblacion.append(list([coste(hijo,alpha), hijo]))
                if(hija.sum()<205):
                    nueva_poblacion.append(list([1000,hija]))
                else:
                    nueva_poblacion.append(list([coste(hija,alpha), hija]))
        nueva_poblacion = sorted(nueva_poblacion, key=lambda x: (x[0]))
        # print(distancia_umbral)
        poblacion=deepcopy(nueva_poblacion[0:30])
        if iguales(nueva_poblacion,poblacion):
            distancia_umbral-=1

    km=poblacion[0][0]-(poblacion[0][1].sum()-205)*alpha
    resultado="fitness: "+str(poblacion[0][0])+"  km: "+str(km)+"  slots: "+str(poblacion[0][1].sum())+"  reinicios: "+str(reinicios)+"  solucion: "+str(poblacion[0][1])
    return resultado

print(chc(30,6))




