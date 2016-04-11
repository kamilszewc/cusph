#!/usr/bin/env python
from sys import argv

f = open(argv[1])
datalines = f.readlines()
f.close()

x = "["
y = "["

for dataline in datalines:
    x += dataline.split(",")[0] + ","
    y += dataline.split(",")[1] + ","

x += "]"
y += "]"

print x
print y
