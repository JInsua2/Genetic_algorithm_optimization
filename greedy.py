import math
from builtins import print

import numpy as np
# import pandas as pd
from random import randint
from itertools import permutations

import time as tim

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


def generar_greedy(sol_act):
    s_ini = sol_act.copy()
    for indice, elem in enumerate(s_ini):
        s_ini[indice] = np.round((elem * 220) / 163)
    return s_ini


inicio3 = tim.time()
greedy = generar_greedy(movInicial)
print(greedy.sum())
# print("la solucion greedy es: ", greedy)
print("el fitness es: ",coste(greedy,alpha=4.5))
print("el tiempo de ", tim.time() - inicio3)
