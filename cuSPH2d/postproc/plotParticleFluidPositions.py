from sys import argv
from numpy import *
import os
from glob import glob
from pylab import *
from postproc.Data import Data

lista = []
for args in argv[1:]:
    for arg in glob(args):
        lista.append(arg)
        
t = []
x = []
y = []
u = []
v = []
        
for arg in lista:
    d = Data(arg)
    t.append(d.t)
    x.append(d.pdpfX[0])
    y.append(d.pdpfY[0])
    u.append(d.pdpfU[0])
    v.append(d.pdpfV[0])
    
figure()
plot(t, x, 'o')
xlabel("t")
ylabel("x")

figure()
plot(t, y, 'o')
xlabel("t")
ylabel("y")

figure()
plot(t, u, 'o')
xlabel("t")
ylabel("u")

figure()
plot(t, v, 'o')
xlabel("t")
ylabel("v")

show()