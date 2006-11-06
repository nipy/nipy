import neuroimaging.data_io.formats.analyze as Ana
import neuroimaging.data_io.formats.ecat7 as Ecat7
import neuroimaging.data_io.formats.nifti1 as Nifti
import numpy as N
from  neuroimaging.core.image.image import Image
from neuroimaging.utils.tests.data import repository

from neuroimaging.utils import wxmpl


myecat = Ecat7.Ecat7("FDG-de.v.bz2",datasource=repository)

#myNifti =  Nifti.Nifti1
#myNifti.intent = 

# init check for input image and create generic outname if one is not provided

# Allow for modification of header fields

# allow for reorientation of image
app = wxmpl.PlotApp('Ecat2Nifti')

fig = app.get_figure()

# Create an Axes on the Figure to plot in.
axes = fig.gca()

# get image
myImg = Image(myecat.data[0])
myImgArray = myImg.readall()
# Plot the Image slice
axes.imshow(myImgArray[22,:,:])
app.MainLoop()


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
        
        self.header['datatype']=Ana.FLOAT
        self.header['dim'] = [3,
                              myecat.data.shape[1],
                              myecat.data.shape[2],
                              myecat.data.shape[3]]
        #self.header[
        
