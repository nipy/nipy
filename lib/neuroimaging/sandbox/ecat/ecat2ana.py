import neuroimaging.data_io.formats.analyze as Ana
import neuroimaging.data_io.formats.ecat7 as Ecat7
import neuroimaging.data_io.formats.nifti1 as Nifti
import numpy as N
from  neuroimaging.core.image.image import Image
from neuroimaging.utils.tests.data import repository

from neuroimaging.utils import wxmpl
import matplotlib.cm as cm

newfile = '/home/surge/cindeem/DEVEL/RAW_PET/B05_206-43D52D9100000211-de.v'
myecat = Ecat7.Ecat7(newfile)
#myecat = Ecat7.Ecat7("FDG-de.v.bz2",datasource=repository)
anaPetFile = '/home/surge/cindeem/DEVEL/RAW_PET/B05_206-43D52D9100000211-de1.hdr'
myAna = Ana.Analyze(anaPetFile)
#myNifti =  Nifti.Nifti1
#myNifti.intent = 

# init check for input image and create generic outname if one is not provided

# Allow for modification of header fields

# allow for reorientation of image
app = wxmpl.PlotApp('Ecat2Nifti')

fig = app.get_figure()

# Create an Axes on the Figure to plot in.
# create 3 axis
axes1 = fig.add_subplot(221)
axes2 = fig.add_subplot(222)
axes3 = fig.add_subplot(223)
axes4 = fig.add_subplot(224)
#axes = fig.gca()

# get image
jnk = myecat.frames[0]
myImg = Image(jnk.data)
myImgArray = myImg.readall()
myImgArray2 = myImgArray.astype('float')/(myecat.scale * jnk.subheader['SCALE_FACTOR'])

myAnaImg =Image(myAna.data)
myAnaArray = myAnaImg.readall()
# Plot the Image slice
#axes1.imshow(myImgArray2[:,128,:]/N.float(myImgArray.max()),
#             cmap=cm.gray)
axes1.imshow(myImgArray2[:,128,:], cmap=cm.gray, origin='lower')

axes1.set_aspect(5,adjustable='box',anchor='C')
axes2.imshow(myImgArray2[:,:,128]/N.float(myImgArray.max()),
             cmap=cm.gray, origin='lower')
axes2.set_aspect(5,adjustable='box',anchor='C')
axes3.imshow(myImgArray2[22,:,:],
             cmap=cm.gray, origin='lower')
axes3.set_aspect(1,adjustable='box',anchor='C')

axes4.plot(myImgArray2[22,128,:])

app.MainLoop()

anafromecat = 


class Ecat2Analyze(Ana.Analyze):
    """
    A class to make a new Analyze instance out of ECAT data
    """
    def __init__(self, myecat, newfilename=myecat.data_file):
        """
        initialize new header
        """
        
    def header_from_given(self):
        #note data type will ned to change to conform to ANALYZE options
        # by default change to Ana.FLOAT

        self.header['bitbix']=32
        isotop = myecat.header.get('isotope_name')[0:9]
        self.header['data_type'] = isotop.replace('\x00','')
        self.header['datatype']=Ana.FLOAT
        self.header['descrip']='ecat2ana.py'
        
        self.header['dim'] = [3,
                              myecat.data.shape[1],
                              myecat.data.shape[2],
                              myecat.data.shape[3]]
        #orientation
        self.header['orient']= self.get_orientation(myecat)
        self.header['origin'] = 
        
    def get_orientation(self,myecat):
        """
        determin original ECAT orientation
        and translate to Analyze format
        """
        ecatorient = myecat.header['patient_orientation']
        if ecatorient == 0:
            anaorient == 0
        else:
            anaorient = -1
        return anaorient

        """
        FFP (0)= FeetFirstProne (face down)
        HFP (1)= HeadFirstProne
        FFS (2)= FeetFirstSupine (face up)
        HFS (3)= HeadFirstSupine
        FFDR (4)= Feet First Decubitus Right (on right side)
        HFDR (5)= HeadFirstDecubitusRight
        FFDL (6)= FeetFirstDecubitusLeft (in left side)
        HFDL (7)= HEadFirstDecubitusLeft
        unknown (8) = unknown
        
        R = right
        L = left
        P = posterior
        A = anterior
        S = superior
        I = inferior

        
        FFP -> RL PA IS -> Ana orient = 0 
        HFP -> LR PA SI -> Ana.orient = -1 (unknown)
        FFS -> LR AP IS -> Ana.orient = -1 (unknown)
        HFS -> RL AP SI -> Ana.orient = -1 (unknown)

        FFDR   PA LR IS -> Ana.orient = -1 (unknown)
        HFDR   AP LR SI -> Ana.orient = -1 (unknown)
        FFDL   AP RL IS -> Ana.orient = -1 (unknown)
        HFDL   PA RL SI -> Ana.orient = -1 (unknown)
        """
