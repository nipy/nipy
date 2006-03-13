import anatomist
# these 3 following imports are only needed for the exaple at the end
import aims
import sys
import os

# This class follows the sample API for an images/meshes viewing window 
# defined during the "coding sprint" in Paris, 20-25 feb. 2006
# Johnatan Tailor
# Yann Cointepas
# Denis Riviere

class SimpleWindow:
    '''Wrapping for an Anatomist window. This kind of view can display 2D, 3D,
    or 4D images, meshes, and many other objects in an arbitrary number in the
    same view Objects can be shared across multiple views.'''

    def __init__(self, vtype='Axial'):
        '''Arguments:
        vtype: view type: 'Axial', 'Coronal', 'Sagittal', '3D', 'Browser',
        'Profile',...'''
        a = anatomist.Anatomist()
        self._anatomist = a
        p = a.theProcessor()
        # open an axial window
        print 'vtype:', vtype
        c = anatomist.CreateWindowCommand(vtype)
        p.execute(c)
        self._window = c.createdWindow() # win is a AWindow object
        self._objects = []

    def __del__(self):
        '''For now the window closes in Anatomist, but objects are not
        destroyed since they may be used in other views.
        To be discussed...'''
        self._anatomist.theProcessor().execute('CloseWindow', 
                                               windows=[self._window])

    def addImage(self, imageData, affineTransformation=None):
        '''Arguments:
        - imageData: image object to view, can be an Anatomist object,
          an aims/carto volume, or a Numeric array (numpy will be supported
          when pyaims uses numpy)
        - affineTransformation (optional): a transformation between the object
          and the view. Not implemented yet.
        Returns:
        The Anatomist object corrsponding to the image (may be imageData if
        imageData is already an Anatomist object'''
        image = anatomist.AObjectConverter.anatomist(imageData)
        c = anatomist.AddObjectCommand([image], [self._window])
        self._anatomist.theProcessor().execute(c)
        self._objects.append(image)
        return image # return Anatomist object as transformed

    def removeImage(self, tag):
        '''Removes an image (or any object, actually) from the view.
        The object is not destroyed in Anatomist
        Arguments:
        - tag: must be an Anatomist object'''
        self._anatomist.theProcessor().execute('RemoveObject', 
                                               objects=[tag], 
                                               windows=[self._window])
        self._objects.remove(tag)

    def addMesh(self, mesh):
        '''A simplified version of addImage. For now, no real difference.
        Arguments:
        - mesh: mesh object, can be an Anatomist or Aims mesh
        Returns:
        The Anatomist object corrsponding to the mesh (may be 'mesh' if
        mesh is already an Anatomist object'''
        # for now, no difference with addImage
        return self.addImage(mesh)

    def setMaterial(self, tag, **material):
        '''Sets the material properties on the given mesh.
        It's rather an object (mesh) method, for me it should not be part
        of the view API, but it's just to show how it works.
        Arguments:
        - tag: the Anatomist object to set the material on
        - **material (kwargs): properties of the material to be set. See the
          example for how to use it.'''
        self._anatomist.theProcessor().execute('SetMaterial', objects=[tag], 
                                               **material)

    def addImageFusion(self, *imageTags):
        '''Makes an images fusion and displays it.
        Arguments:
        - imageTags (list args): images to fusion. More than 2 images are
          allowed. The same restrictions on image types apply than for the
          addImage method
        Returns:
        The fusion object, which is an Anatomist object'''
        images = [anatomist.AObjectConverter.anatomist( x ) \
                  for x in imageTags]
        p = self._anatomist.theProcessor()
        # set a different palette on the second object so we can see something
        if len(images) >= 2:
            c = anatomist.SetObjectPaletteCommand([images[1]], 
                                                  'Blue-Red-fusion')
            p.execute(c)
        c = anatomist.FusionObjectsCommand(images, 'Fusion2DMethod')
        p.execute(c)
        fus = c.createdObject() # this is the fusionned object
        c = anatomist.AddObjectCommand([fus], [self._window])
        p.execute(c)
        self._objects.append(fus)
        return fus

# Mesh creation

# ISO surface
def createIsoSurface(image, threshold):
    'Not implemented yet.'
    pass

  
## test example part (should probably move to another example file)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage:', sys.argv[0], '<image_path>'
        print '<image_path> should refer to a directory containing:'
        print '- an anatomical MRI volume named irm.ima/irm.dim (GIS format)'
        print '- a functional volume named spect.ima/spect.dim'
        print '- a mesh named ra_head.mesh'
        sys.exit(1)
    imagepath = sys.argv[1]
    # load any volume as a aims.Volume_* object
    r = aims.Reader()
    vol = r.read(os.path.join(imagepath, 'irm.ima'))
    # initialize Anatomist
    a = anatomist.Anatomist()
    # open an axial window
    view = SimpleWindow()
    # put volume in window
    view.addImage( vol )

    # another volume
    vol2 = r.read(os.path.join(imagepath, 'spect.ima'))
    #fusion with vol
    view2 = SimpleWindow()
    view2.addImageFusion(vol, vol2)

    # now a mesh
    m = r.read(os.path.join(imagepath, 'ra_head.mesh'))
    view3 = SimpleWindow()
    am = view3.addMesh(m)
    view3.setMaterial(am, diffuse=[0., 0., 1., 1.])
