import numpy as np
import sys
sys.path.append("../")
import base as base

"""
Lectura de parametros de la linea de comandos, el numero de pares de moleculas 
en la cadena
"""
L= float(sys.argv[1])
inter= float( sys.argv[2] )


"""
Variables de los campos magneticos en Z y X
"""
hz_list = np.linspace(0.0, 2.0, 31)
hx_list = np.linspace(0.0, 0.2, 21)
hz_list[0] = 0.01

for hx in hz_list:
    for hz in hz_list:
        #Ejecucion de DMRG bajo los parametros ingresados
        base.main(J=1.49, J13=-0.89, Jinter=inter, hz=hz, hx=hx, L=L, model="linear")
