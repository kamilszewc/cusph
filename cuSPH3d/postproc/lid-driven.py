#!/usr/bin/env python
from sys import argv
from numpy import *
from math import *
from matplotlib.pyplot import *
from postproc.data import data

def kernel(q):
    return (42.0/256.)*(q+0.5)*(2.0-q)**4

fig1 = figure(figsize=(6.5,6.5)); plt1 = fig1.add_subplot(111)
fig2 = figure(figsize=(6.5,6.5)); plt2 = fig2.add_subplot(111)

d = data(argv[1], ("velocity", "mass", "density"))

ls = linspace(2.0*d.H, 1.0-2.0*d.H, 100)
ws = []
us = []
for l in ls:
    value_w = 0.0
    xc = long( 0.5*l/d.H )
    yc = long( 0.5*0.5/d.H )
    zc = long( 0.5*0.5/d.H )
    c = xc + yc*d.NXC + zc*d.NXC*d.NYC
    
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                cellNumber = (xc+i) + (yc+j)*d.NXC + (zc+k)*d.NXC*d.NYC
                try:
                    for pid in d.particleInCell[cellNumber]:
                        r2 = (d.x[pid]-l)**2 +(d.y[pid]-0.5)**2 +(d.z[pid]-0.5)**2
                        if r2 < 4.0*d.H*d.H:
                            value_w += d.w[pid] * d.KNORM * kernel( sqrt(r2) * d.I_H ) * d.m[pid] / d.d[pid]
                except:
                    pass
    ws.append(float(value_w))

    
for l in ls:
    value_u = 0.0
    xc = long( 0.5*0.5/d.H )
    yc = long( 0.5*0.5/d.H )
    zc = long( 0.5*l/d.H )
    c = xc + yc*d.NXC + zc*d.NXC*d.NYC
    
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                cellNumber = (xc+i) + (yc+j)*d.NXC + (zc+k)*d.NXC*d.NYC
                try:
                    for pid in d.particleInCell[cellNumber]:
                        r2 = (d.x[pid]-0.5)**2 +(d.y[pid]-0.5)**2 +(d.z[pid]-l)**2
                        if r2 < 4.0*d.H*d.H:
                            value_u += d.u[pid] * d.KNORM * kernel( sqrt(r2) * d.I_H ) * d.m[pid] / d.d[pid]
                except:
                    pass
    us.append(float(value_u))
    
    #for i in range(d.N):
    #    r2 = (d.x[i]-0.5)**2 +(d.y[i]-0.5)**2 +(d.z[i]-l)**2
    #    if r2 < 4.0*d.H*d.H:
    #        value_u += d.u[i] * d.KNORM * kernel( sqrt(r2) * d.I_H ) * d.m[i] / d.d[i]
    #us.append(float(value_u))

plt1.plot(ls, us, "k-", label="SPH", linewidth=1.5)
plt2.plot(ls, ws, "k-", label="SPH", linewidth=1.5) 



if len(argv) == 3 and argv[2] == "100":
    for filename, label, marker in zip( ("resources/lid-driven-3d-v-shu.dat", "resources/lid-driven-3d-v-lo.dat"), ("Shu et al.","Lo et al."), ('ok', 'sk') ):
        file = open(filename, 'r')
        datalines = file.readlines()
        file.close()
        refxw, refw = [], []
        for dataline in datalines:
            refxw.append(float(dataline.split()[0]))
            refw.append(float(dataline.split()[1]))
        plt2.plot(refxw, refw, marker, label=label)
        plt2.set_ylabel(r"$u_z/|{\bf u}_w|$", fontsize='x-large')
        plt2.set_xlabel(r"$x/L$", fontsize='x-large')

    for filename, label, marker in zip( ("resources/lid-driven-3d-u-shu.dat", "resources/lid-driven-3d-u-lo.dat"), ("Shu et al.", "Lo et al."), ('ok', 'sk') ):
        file = open(filename, 'r')
        datalines = file.readlines()
        file.close
        refxu, refu = [], []
        for dataline in datalines:
            refxu.append(float(dataline.split()[0]))
            refu.append(float(dataline.split()[1]))
        plt1.plot(refxu, refu, marker, label=label)
        plt1.set_ylabel(r"$u_x/|{\bf u}_w|$", fontsize='x-large')
        plt1.set_xlabel(r"$y/L$", fontsize='x-large')
elif len(argv) == 3 and argv[2] == "1000":
    for filename, label, marker in zip( ("resources/lid-driven-3d-v-yang.dat",), ("Yang et al.",), ('ok',) ):
        file = open(filename, 'r')
        datalines = file.readlines()
        file.close()
        refxw, refw = [], []
        for dataline in datalines:
            refxw.append(float(dataline.split(',')[0]))
            refw.append(float(dataline.split(',')[1]))
        plt2.plot(refxw, refw, marker, label=label)
        plt2.set_ylabel(r"$u_z/|{\bf u}_w|$", fontsize='x-large')
        plt2.set_xlabel(r"$x/L$", fontsize='x-large')

    for filename, label, marker in zip( ("resources/lid-driven-3d-u-yang.dat",), ("Yang et al.",), ('ok',) ):
        file = open(filename, 'r')
        datalines = file.readlines()
        file.close
        refxu, refu = [], []
        for dataline in datalines:
            refxu.append(float(dataline.split(',')[0]))
            refu.append(float(dataline.split(',')[1]))
        plt1.plot(refxu, refu, marker, label=label)
        plt1.set_ylabel(r"$u_x/|{\bf u}_w|$", fontsize='x-large')
        plt1.set_xlabel(r"$y/L$", fontsize='x-large')
elif len(argv) == 3 and argv[2] == "10000":
    for filename, label, marker in zip( ("resources/lid-driven-3d-koseff-street-v-wall.dat",), ("Koseff and Street",), ('ok',) ):
        file = open(filename, 'r')
        datalines = file.readlines()
        file.close()
        refxw, refw = [], []
        for dataline in datalines:
            refxw.append(float(dataline.split(',')[0]))
            refw.append(float(dataline.split(',')[1]))
        plt2.plot(refxw, refw, marker, label=label)
        plt2.set_ylabel(r"$u_z/|{\bf u}_w|$", fontsize='x-large')
        plt2.set_xlabel(r"$x/L$", fontsize='x-large')

    for filename, label, marker in zip( ("resources/lid-driven-3d-koseff-street-u-wall.dat",), ("Koseff and Street",), ('ok',) ):
        file = open(filename, 'r')
        datalines = file.readlines()
        file.close
        refxu, refu = [], []
        for dataline in datalines:
            refxu.append(float(dataline.split(',')[0]))
            refu.append(float(dataline.split(',')[1]))
        plt1.plot(1.0-array(refxu), refu, marker, label=label)
        plt1.set_ylabel(r"$u_x/|{\bf u}_w|$", fontsize='x-large')
        plt1.set_xlabel(r"$y/L$", fontsize='x-large')

plt1.legend()
plt2.legend()

plt1.grid(True)
plt2.grid(True)

plt1.set_xlim(0.0, 1.0)
plt2.set_xlim(0.0, 1.0)

show()

