#!/usr/bin/env python
from sys import *

f = file(argv[1])
datalines = f.readlines()
f.close()

for dataline in datalines:
    for word in dataline.split():
        if word == "inf":
            print word
#        if word == "nan":
#            print dataline.split[0,3] 
