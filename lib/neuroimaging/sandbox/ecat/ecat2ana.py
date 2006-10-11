import neuroimaging.data_io.formats.analyze as Ana
import neuroimaging.data_io.formats.ecat7 as Ecat7
import numpy as N
from  neuroimaging.core.image.image import Image
from neuroimaging.utils.tests.data import repository

## newfile = '/home/surge/cindeem/DEVEL/TestData/nitest.img'
## from neuroimaging.core.image.image import Image
## myImg = Image(tmpdat)
## myImg.tofile(newfile)
## newfile = '/home/surge/cindeem/DEVEL/TestData/nitest.hdr'
## myImg.tofile(newfile)
## newImg = myImg.toarray()
## newImg = Analyze.Analyze(newImg)

myecat = Ecat7.Ecat7("FDG-de.v",datasource=repository)


jnk = Ecat7.CacheData('anaTest1.img')
jnkname = jnk.filename('anaTest1.img')

jnkhdr = Ecat7.CacheData('anaTest1.hdr')
jnkhdrname = jnkhdr.filename('anaTest1.hdr')

tmphdr =open(jnkhdrname,'w')
tmphdr.close()

tmpdat = myecat.frames[0]

tmpdat.data.tofile(jnkname)
myArray = N.array(tmpdat.data, tmpdat.sctype)
myArray.tofile(jnkname)
#myArray.tofile(jnkhdrname)

myImg = Image(jnkhdrname)

#myAna = Ana.Analyze(myArray)


