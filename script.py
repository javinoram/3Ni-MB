import numpy as np
import sys

sys.path.append("..")
import perpendicular.base as base

hz_list = np.linspace(0.0, 2.0, 41)
hz_list[0] = 0.01
for hz in hz_list:
    base.main(Jinter=-0.25, hz=hz, hx=0.0, L=3, model="linear")
