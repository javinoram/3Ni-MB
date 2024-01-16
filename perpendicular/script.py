import numpy as np
import sys

sys.path.append("..")
import perpendicular.base as base

L= float(sys.argv[1])
inter= float( sys.argv[2] )

hz_list = np.linspace(0.0, 2.0, 31)
hx_list = np.linspace(0.0, 0.2, 21)
hz_list[0] = 0.01

for hx in hz_list:
    for hz in hz_list:
        base.main(J=1.49, J13=-0.89, Jinter=inter, hz=hz, hx=hx, L=L, model="parallel")
