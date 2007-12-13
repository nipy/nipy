"""
A simple example of ipython1 parallel applications.

To run this, we must first create some ipython engines.
The following command will create 10 engines

ipcluster -n 10

This example computes the MAD across time for an fMRI volume
by splitting the slices over the ipython engines.

"""

import ipython1.kernel.api as kernel
ipc = kernel.RemoteController(('127.0.0.1',10105))

ipc.executeAll(file("header.py").read())
ipc.executeAll(file("mad.py").read())
ipc.scatterAll("slices", range(30))
ipc.executeAll("""
f = Image('http://kff.stanford.edu/FIAC/fiac3/fonc3/fsl/filtered_func_data.img')
m = []
print f.shape, slices
for s in slices:
    m.append(MAD(f[:,s]))
""")

slices = ipc.gatherAll("slices")
m = ipc.gatherAll("m")

from fiac import anat
import numpy as N
mm = N.array([m[s] for s in slices])

print mm.shape
