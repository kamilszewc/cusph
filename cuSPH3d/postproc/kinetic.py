#!/usr/bin/env python

from sys import argv
from matplotlib.pylab import *

if __name__ == '__main__':
    try:
        file = open(argv[1], "r")
    except:
        print("Error: there is no %s file" % argv[1])

    datalines = file.readlines()
    file.close()

    t, k = [], []
    for dataline in datalines:
        t.append(float(dataline.split()[0]))
        k.append(float(dataline.split()[1]))

    figure()
    plot(t, k)
    xlabel(r"$t$", fontsize='x-large')
    ylabel(r"$E_k$", fontsize='x-large')
    title("Kinetic energy")

    figure()
    semilogy(t, k)
    xlabel(r"$t$", fontsize='x-large')
    ylabel(r"$\log E_k$", fontsize='x-large')
    title("Kinetic energy")

    show()
