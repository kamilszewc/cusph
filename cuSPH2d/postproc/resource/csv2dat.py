#!/usr/bin/env python
from sys import argv

f = open(argv[1])
datalines = f.readlines()
f.close()

for dataline in datalines:
    x = float(dataline.split(",")[0])
    y = float(dataline.split(",")[1])
    print x, y
