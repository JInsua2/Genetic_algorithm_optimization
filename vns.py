import math
import random
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


def coste(solucion_actual):
    capacidadEstaciones = solucion_actual.copy()
    huecosLibres = capacidadEstaciones.copy()
    bicisEstacion = np.array(capacidadEstaciones - huecosLibres)
    kmaux = 0
    for x in range(0, 16):
        ejecutar_accion(movInicial[x], x, capacidadEstaciones, huecosLibres, bicisEstacion)

    for elem in lista_mov:
        kmaux += ejecutar_accion(elem[1], elem[0], capacidadEstaciones, huecosLibres, bicisEstacion)

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

def genera_vecino_vns(sub_vector,k):
    x=0
    while x<k:

        posicion=randint(0,len(sub_vector))
        posicion2=randint(0,len(sub_vector))
        #comprobar q  sean distintos y cambiar los valores

def mutacion(s_ini,k):
    k=k*4

    # posicion=randint(0,len(s_ini)-1)
    posicion=15
    posiciones=[]
    sub_vector=[]

    for x in range(0,k):
        pos=posicion % 16
        posiciones.append(pos)
        sub_vector.append(s_ini[pos])
        posicion+=1
    mutacion=genera_vecino_vns(sub_vector)
    j=0
    for valor in mutacion:
        s_ini[posiciones[j]]=valor
        j+=1
    return s_ini
def genera_vecino_vns(lista):
    r=randint(0,len(lista)-1)
    r2=randint(0,len(lista)-1)
    if (lista[r] > 0):
        lista[r]-=2
        lista[r2]+=2
    else:
        lista[r2]-=2
        lista[r]+=2
    return lista

def vns(solucion):
    k = 1
    kmax = 4
    s_mejor = solucion.copy()
    coste_mejor = coste(s_mejor)
    s_mutacion=solucion.copy

    while (k <= kmax):
        s_local = busqueda_local(solucion)
        coste_local = coste(s_local)
        if coste_local < coste_mejor:
            s_mejor = s_local
            coste_mejor = coste_local
            k = 1
        else:
            k += 1
        s_mutacion=mutacion(s_mejor,k)

    return s_mejor


actual = randomlist(16, 220)
resultado = vns(actual)
print("el resultado de local es : ", resultado,"  ",resultado.sum())
print("el coste es: ", coste(resultado))
