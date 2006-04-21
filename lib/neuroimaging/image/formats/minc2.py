#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW
#log# opts = Struct({'logfile': 'ipython_log.py'})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
import tables
minc2 = tables.openFile('avg152T1-2.mnc')
ipalias("ls ")
minc2 = tables.openFile('avg152T1-2(2).mnc')
minc2
import numpy
i = numpy.asarray(minc2.getNode('/minc-2.0/image/0/image'))
i
import matplotlib
import pylab
pylab.mshow(i[10,...])
pylab.imshow(i[10,...])
pylab.show()
imax = numpy.asarray(minc2.getNode('/minc-2.0/image/0/image-max'))
imin = numpy.asarray(minc2.getNode('/minc-2.0/image/0/image-min'))
imax
imin
i.size()
i.shape()
i.size
i.shape
imin.shape
ipmagic("hist -n")
help(pylab.imshow)
pylab.imshow(i[10,...], interpolation='nearest')
i[10,...]
i.max()
i.min()
ipmagic("hist -n")
ipmagic("%log start")
ipmagic("%logon ")
ipmagic("%logstart ")

ipalias("cat ipython_log.py")
imin
imax
i = numpy.asarray(minc2.getNode('/minc-2.0/image/0/image')[10,...])
i
#?minc2.getNode
n = minc2.getNode('/minc-2.0/image/0/image')
#?n.read
test = n.read((1,1,1), (4, 4, 4))
test = n.read(1, 4)
test
test.shape()
test.shape
n = minc2.getNode('/minc-2.0/image/0/image')
type(n)
n[0]
n.shape
n[[slice(0,10,2), slice(0,10,2), slice(3,15,3)]]
n[slice(0,10,2), slice(0,10,2), slice(3,15,3)]
n[slice(0,10,2), slice(0,10,2), slice(3,15,3)].shape
x=numpy.zeros((40,)*3)
x[[slice(0,10,2), slice(0,10,2), slice(3,15,3)]]
n.__getitem__([slice(0,10,2), slice(0,10,2), slice(3,15,3)])
n.__getitem__(slice(0,10,2), slice(0,10,2), slice(3,15,3))
n[[slice(0,10,2), slice(0,10,2), slice(3,15,3)]]
