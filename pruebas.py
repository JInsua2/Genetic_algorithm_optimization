import random as rn
import time

import numpy as np
from random import randint

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def randomlist(m, n):
    arr = np.zeros(m)
    for i in range(n):
        valor = randint(0, n)
        aux = valor % (m)
        arr[aux] += 1
    return arr


def cruce_blx_padres(padre, madre):
    numeros = np.arange(0, 15)
    posiciones = rn.choices(numeros, k=6)
    hijo = padre.copy()
    hija = madre.copy()
    print(posiciones)
    for pos in posiciones:
        hija[pos] = padre[pos]
        hijo[pos] = madre[pos]
    return hijo, hija
def distancia_hamming(padre,madre):
    distancia=0
    indices=[]
    for i,(p,m) in enumerate(zip(padre,madre)):
        if p!=m:
            distancia+=1
            indices.append(i)
    return indices,distancia

padre=[0,1,3,4]
madre=[0,1,2,3]
indices,distancia=distancia_hamming(padre,madre)

# padre = randomlist(16, 220)
# madre = randomlist(16, 220)
# print(padre, "   padres ", madre)
# print(distancia_hamming(padre,madre))
# hijo, hija = cruce(padre, madre)
# print(hijo, "  hijos   ", hija)
# print(hijo.sum(), "      ", hija.sum())

a=np.array([18., 12., 19., 15., 20., 11., 12.,  9., 12., 13., 17., 26.,  3.,12., 18., 12.]).sum()
b=a-205
print(b*6)