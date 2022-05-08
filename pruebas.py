import random as rn
import time

import numpy as np
from random import randint




import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

# Initialize communication with TMP102

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

    # Read temperature (Celsius) from TMP102
    temp_c = randint(0,300)

    # Add x and y to lists
    xs.append(time.time())
    ys.append(temp_c)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    # plt.axis([0, 100,0, 100])
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('coste km sobre el tiempo')
    plt.ylabel('Km')

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
plt.show()




# def randomlist(m, n):
#     arr = np.zeros(m)
#     for i in range(n):
#         valor = randint(0, n)
#         aux = valor % (m)
#         arr[aux] += 1
#     return arr
#
# def genera_poblacion_inicial():
#     nueva_poblacion=[]
#     for x in range(0,25):
#         aleatorio=randomlist(16,220)
#         neuvo_elem=list([159,aleatorio])
#         nueva_poblacion.append(neuvo_elem)
#     return nueva_poblacion
# poblacion=genera_poblacion_inicial()
# # pobla=list(poblacion)
# print(type(poblacion[5]))
# poblacion[5][0]=123
# print(poblacion[5])
#
# # hijo=padre[0:2]
# # hijo=np.concatenate((padre[0:2],padre[3:4],padre[4:5]),axis=0)
# # hijo+=padre[3:6]
# prueba=[1,2,3,2,2,0]
# prueba.sort()
# print(prueba)
#
# # hijo+=padre[6:7]
# # hijo.append(hijo2)
