import numpy as np
import sys
sys.path.append("..")
from base import *

"""
Lectura de parametros de la linea de comandos, el numero de pares de moleculas 
en la cadena
"""
L= float(sys.argv[1])

"""
Variables de los campos magneticos en Z y X
"""
hz_list = np.linspace(0.0, 2.0, 31)
hx_list = np.linspace(0.0, 0.25, 11)
hz_list[0] = 0.01

for hx in hz_list:
    for hz in hz_list:
        #Ejecucion de DMRG bajo los parametros ingresados
        main(J=1.49, J13=-0.89, Jinter=0.0, hz=hz, hx=hx, L=L, model="parallel")
