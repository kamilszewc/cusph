#!/usr/bin/env python

import os
import re
import lzma
import hashlib

files = os.listdir("results")
files = filter( (lambda files: re.match(r"\d{1,}.\d{6}.xml$", files)), files)

for file in files:
    if file.split(".")[-1] == 'dat':
        print("Archivize %s" % (file))
        try:
            f = open("results/" + file, "r")
            data = f.read()
            f.close()
        except:
            print("Error reading files")
            exit()

        try:
            f = lzma.LZMAFile("results/" + file + ".xz", "w")
            f.write(data)
            f.close()
        except:
            print("Error writing files")
            exit()

        try:
            os.remove("results/" + file)
        except:
            print("Error removing files")

        try:
            shanum = hashlib.sha256()
            shanum.update(data) 
            f = open("results/" + "shasum.dat", "a")
            f.write("%s\t%s" % (file, shanum.hexdigest()) )
            f.close()
        except:
            print("Error creating sha sum")

