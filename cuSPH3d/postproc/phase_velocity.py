#!/usr/bin/env python
from sys import argv
from postproc.data import data

for arg in argv[1:]:
    d = data(argv[1])

    vel_x, vel_y, vel_z, pos_x, pos_y, pos_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n = 0
    for x, y, z, u, v, w, c in zip(d.x, d.y, d.z, d.u, d.v, d.w, d.c):
        if c > 0.5:
            pos_x += x
            pos_y += y
            pos_z += z
            vel_x += u
            vel_y += v
            vel_z += w
            n += 1

    print pos_x/n, pos_y/n, pos_z/n, vel_x/n, vel_y/n, vel_z/n 
