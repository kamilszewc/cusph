#!/usr/bin/env python

import os
import re
import bz2
import hashlib
from multiprocessing import Process

N = 6 
SLEEP_TIME = 1800

def arch(files, i):
    for file in files:
        if file.split(".")[-1] == 'dat':
            print "Archivize %s in process %s" % (file, i)
            try:
                f = open("results/" + file, "r")
                data = f.read()
                f.close()
            except:
                print "Error reading files"
                exit()

            try:
                f = bz2.BZ2File("results/" + file + ".bz2", "w")
                f.write(data)
                f.close()
            except:
                print "Error writing files"
                exit()

            try:
                os.remove("results/" + file)
            except:
                print "Error removing files"

while (True):
    os.system("sleep %s" % (SLEEP_TIME,))

    allfiles = os.listdir("results")
    allfiles = filter( (lambda allfiles: re.match(r"\d{1,}.\d{6}.dat$", allfiles)), allfiles)

    n = len(allfiles)
    m = (n+N-1)/N
    files = []
    num = 0
    for i in range(N):
        if num+m < n:
            files.append(allfiles[num:num+m])
        else:
            files.append(allfiles[num:])
        num += m

    ps = []
    for i in range(N):
        ps.append( Process(target=arch, args=(files[i], i)) )

    for i in range(N):
        ps[i].start()

    for i in range(N):
        ps[i].join()

