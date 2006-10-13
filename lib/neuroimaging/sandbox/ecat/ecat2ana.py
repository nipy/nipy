import neuroimaging.data_io.formats.analyze as Ana
import neuroimaging.data_io.formats.ecat7 as Ecat7
import numpy as N
from  neuroimaging.core.image.image import Image

pfile = '/home/surge/cindeem/DEVEL/RAW_PET/B05_206-43D52D9100000211-de.v'

myecat = Ecat7.Ecat7(pfile)


jnk = Ecat7.CacheData('anaTest1.img')
jnkname = jnk.filename('anaTest1.img')

jnkhdr = Ecat7.CacheData('anaTest1.hdr')
jnkhdrname = jnkhdr.filename('anaTest1.hdr')

tmphdr =open(jnkhdrname,'w')
tmphdr.close()

tmpdat = myecat.frames[0]

tmpdat.data.tofile(jnkname)
myArray = N.array(tmpdat.data, tmpdat.sctype)

#myArray.tofile(jnkhdrname)

myImg = Image(myArray)
myImg.tofile(jnkname)
#myAna = Ana.Analyze.write_header('jnkhdrname')

myAna = Ana.Analyze(jnkname)
