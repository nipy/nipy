import neuroimaging.data_io.formats.analyze as Ana
import neuroimaging.data_io.formats.ecat7 as Ecat7
import neuroimaging.data_io.formats.nifti1 as Nifti
from neuroimaging.data_io.formats import utils
import numpy as N

from neuroimaging.utils import wxmpl
import matplotlib.cm as cm





class Ecat2Analyze(Ana.Analyze):
    """
    A class to make a new Analyze instance out of ECAT data
    """
    def __init__(self, myecat, newfilename,frame = 0):
        """
        initialize new header
        """
        
        tmpframe = myecat.frames[frame]
        
        #Ana.Analyze.__init__(self, newfilename, mode='r')
        self.header_formats = Ana.struct_formats
        Ana.Analyze.header_defaults()
        self.header_from_given()

        
        self.grid = tmpframe.grid
        self.dtype = tmpframe.data.dtype
        self.byteorder = utils.NATIVE
        self.dtype = self.dtype.newbyteorder(self.byteorder)

        self.data = tmpframe.postread(tmpframe.data)
        
        
        #Ana.Analyze.__init__(self, newfilename, mode='w')

        

        
        
    def header_from_given(self,myecat,tmpframe):
        #note data type will need to change to conform to ANALYZE options
        # by default change to Ana.FLOAT

        

        self.header['bitbix']=32
        isotop = self.myecat.header.get('isotope_name')[0:9]
        self.header['data_type'] = isotop.replace('\x00','')
        self.header['datatype']=Ana.FLOAT
        self.header['descrip']='ecat2ana.py'
        
        self.header['dim'] = [3,
                              self.myecat.data.shape[1],
                              self.myecat.data.shape[2],
                              self.myecat.data.shape[3]]
        #orientation
        self.header['orient']= self.get_orientation(myecat)
        self.header['origin'] =[0.0 + tmpframe.subheader.get('X_OFFSET'),
                                0.0 + tmpframe.subheader.get('Y_OFFSET'),
                                0.0 + tmpframe.subheader.get('Z_OFFSET')]
        self.header['pixdim'] = [1,tmpframe.subheader.get('X_PIXEL_SIZE'),
                                 tmpframe.subheader.get('Y_PIXEL_SIZE'),
                                 tmpframe.subheader.get('Z_PIXEL_SIZE'),
                                 0.0,0.0,0.0,0.0]
        self.header['scale_factor'] = 1.0
        # will account for scale factor in acutal data
        
        
                                 

        

        
        
    def get_origin(self, myecat):
        """
        determin original origin from ecat
        """
        
        
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


newfile = '/home/surge/cindeem/DEVEL/RAW_PET/B05_206-43D52D9100000211-de.v'
myecat = Ecat7.Ecat7(newfile)

newfileA = '/home/surge/cindeem/DEVEL/TestData/newAna.hdr'
newAna = Ecat2Analyze(myecat, newfileA,frame=0) 
