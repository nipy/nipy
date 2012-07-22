.. _coordmap-discussion:

########################################
Some discussion notes on coordinate maps
########################################

These notes contain some email discussion between Jonathan Taylor, Bertrand
Thirion and Gael Varoquaux about coordinate maps, coordinate systems and
transforms.

They are a little bit rough and undigested in their current form, but they might
be useful for background.

The code and discussion below mentions ideas like ``LPIImage``, ``XYZImage`` and
``AffineImage``.  These were image classes that constrained their coordinate
maps to have input and output axes in a particular order.  We eventually removed
these in favor of automated reordering of image axes on save, and explicit
reordering of images that needed known axis ordering.

.. some working notes

    import sympy
    i, j, k = sympy.symbols('i, j, k')
    np.dot(np.array([[0,0,1],[1,0,0],[0,1,0]]), np.array([i,j,k]))
    kij = CoordinateSystem('kij')
    ijk_to_kij = AffineTransform(ijk, kij, np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]))
    ijk_to_kij([i,j,k])
    kij = CoordinateSystem('kij')
    ijk_to_kij = AffineTransform(ijk, kij, np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]))
    ijk_to_kij([i,j,k])
    kij_to_RAS = compose(ijk_to_kij, ijk_to_RAS)
    kij_to_RAS = compose(ijk_to_RAS,ijk_to_kij)
    kij_to_RAS = compose(ijk_to_RAS,ijk_to_kij.inverse())
    kij_to_RAS
    kij = CoordinateSystem('kij')
    ijk_to_kij = AffineTransform(ijk, kij, np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]))
    # Check that it does the right permutation
    ijk_to_kij([i,j,k])
    # Yup, now let's try to make a kij_to_RAS transform
    # At first guess, we might try
    kij_to_RAS = compose(ijk_to_RAS,ijk_to_kij)
    # but we have a problem, we've asked for a composition that doesn't make sense
    kij_to_RAS = compose(ijk_to_RAS,ijk_to_kij.inverse())
    kij_to_RAS
    # check that things are working -- I should get the same value at i=20,j=30,k=40 for both mappings, only the arguments are reversed
    ijk_to_RAS([i,j,k])
    kij_to_RAS([k,i,j])
    another_kij_to_RAS = ijk_to_RAS.reordered_domain('kij')
    another_kij_to_RAS([k,i,j])
    # rather than finding the permuation matrix your self
    another_kij_to_RAS = ijk_to_RAS.reordered_domain('kij')
    another_kij_to_RAS([k,i,j])

    >>> ijk = CoordinateSystem('ijk', coord_dtype=np.array(sympy.Symbol('x')).dtype)
    >>> xyz = CoordinateSystem('xyz', coord_dtype=np.array(sympy.Symbol('x')).dtype)
    >>> x_start, y_start, z_start = [sympy.Symbol(s) for s in ['x_start', 'y_start', 'z_start']]
    >>> x_step, y_step, z_step = [sympy.Symbol(s) for s in ['x_step', 'y_step', 'z_step']]
    >>> i, j, k = [sympy.Symbol(s) for s in 'ijk']
    >>> T = np.array([[x_step,0,0,x_start],[0,y_step,0,y_start],[0,0,z_step,z_start],[0,0,0,1]])
    >>> T
    array([[x_step, 0, 0, x_start],
        [0, y_step, 0, y_start],
        [0, 0, z_step, z_start],
        [0, 0, 0, 1]], dtype=object)
    >>> A = AffineTransform(ijk, xyz, T)
    >>> A
    AffineTransform(
    function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=object),
    function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=object),
    affine=array([[x_step, 0, 0, x_start],
                    [0, y_step, 0, y_start],
                    [0, 0, z_step, z_start],
                    [0, 0, 0, 1]], dtype=object)
    )
    >>> A([i,j,k])
    array([x_start + i*x_step, y_start + j*y_step, z_start + k*z_step], dtype=object)
    >>> # this is another 
    >>> A_kij = A.reordered_domain('kij')

    >>> A_kij
    AffineTransform(
    function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='', coord_dtype=object),
    function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=object),
    affine=array([[0, x_step, 0, x_start],
                    [0, 0, y_step, y_start],
                    [z_step, 0, 0, z_start],
                    [0.0, 0.0, 0.0, 1.0]], dtype=object)
    )
    >>>
    >>> A_kij([k,i,j])
    array([x_start + i*x_step, y_start + j*y_step, z_start + k*z_step], dtype=object)
                                                                                    >>> # let's look at another reordering
    >>> A_kij_yzx = A_kij.reordered_range('yzx')
    >>> A_kij_yzx
    AffineTransform(
    function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='', coord_dtype=object),
    function_range=CoordinateSystem(coord_names=('y', 'z', 'x'), name='', coord_dtype=object),
    affine=array([[0, 0, y_step, y_start],
                    [z_step, 0, 0, z_start],
                    [0, x_step, 0, x_start],
                    [0, 0, 0, 1.00000000000000]], dtype=object)
    )
    >>> A_kij_yzx([k,i,j])
    array([y_start + j*y_step, z_start + k*z_step, x_start + i*x_step], dtype=object)
    >>>

    class RASTransform(AffineTransform):
    """
    An AffineTransform with output, i.e. range:

    x: units of 1mm increasing from Right to Left
    y: units of 1mm increasing from Anterior to Posterior
    z:  units of 1mm increasing from Superior to Inferior
    """
    def reorder_range(self):
        raise ValueError('not allowed to reorder the "xyz" output coordinates')

    def to_LPS(self):
        from copy import copy
        return AffineTransform(copy(self.function_domain),
                                copy(self.function_range),
                                np.dot(np.diag([-1,-1,1,1], self.affine))

    class LPSTransform(AffineTransform):
    """
    An AffineTransform with output, i.e. range:

    x: units of 1mm increasing from Left to Right
    y: units of 1mm increasing from Posterior to Anterior
    z:  units of 1mm increasing from Inferior to Superior
    """
    def reorder_range(self):
        raise ValueError('not allowed to reorder the "xyz" output coordinates')


    def to_RAS(self):
        from copy import copy
        return AffineTransform(copy(self.function_domain),
                                copy(self.function_range),
                                np.dot(np.diag([-1,-1,1,1], self.affine)))

    class NeuroImage(Image):
    def __init__(self, data, affine, axis_names, world='world-RAS'):
        affine_transform = {'LPS':LPSTransform,
                            'RAS':RAITransform}[world])(axis_names[:3], "xyz", affine}
        ...

    LPIImage only forced it to be of one type.

Email #1
--------

Excuse the long email but I started writing, and then it started looking like documentation. I will put most of it into doc/users/coordinate_map.rst.


    Also, I am not sure what this means. The image is in LPI ordering, only
    if the reference frame of the world space it is pointing to is. 


I am proposing we enforce the world space to have this frame of reference
to be explicit so that you could tell left from right on an image after calling xyz_ordered().


    If it is
    pointing to MNI152 (or Talairach), then x=Left to Right, y=Posterior to
    Anterior, and z=Inferior to Superior. If not, you are not in MNI152.
    Moreover, according to the FSL docs, the whole 'anatomical' versus
    'neurological' mess that I hear has been a long standing problem has
    nothing to do with the target frame of reference, but only with the way
    the data is stored.


I think the LPI designation simply specifies "x=Left to Right, y=Posterior to
Anterior, and z=Inferior to Superior" so any MNI152 or Tailarach would be in LPI
coordinates, that's all I'm trying to specify with the designation "LPI". If
MNI152 might imply a certain voxel size, then I would prefer not to use MNI152.

If there's a better colour for the bike shed, then I'll let someone else paint it, :)

This LPI specification actually makes a difference to the
"AffineImage/LPIImage.xyz_ordered" method. If, in the interest of being
explicit, we would enforce the direction of x,y,z in LPI/Neuro/AffineImage, then
the goal of having "xyz_ordered" return an image with an affine that has a
diagonal with positive entries, as in the AffineImage specification, means that
you might have to call

affine_image.get_data()[::-1,::-1] # or some other combination of flips

(i.e. you have to change how it is stored in memory).

The other way to return an diagonal affine with positive entries is to flip send
x to -x, y to -y, i.e. multiply the diagonal matrix by np.diag([-1,-1,1,1]) on
the left. But then your AffineImage would now have "x=Right to Left, y=Anterior
to Posterior" and we have lost the interpretation of x,y,z as LPI coordinates.

By being explicit about the direction of x,y,z we know that if the affine matrix
was diagonal and had a negative entry in the first position, then we know that
left and right were flipped when viewed with a command like::

    >>> pylab.imshow(image.get_data()[:,:,10])

Without specifying the direction of x,y,z we just don't know.

    You can of course create a new coordinate system describing, for instance
    the scanner space, where the first coordinnate is not x, and the second
    not y, ... but I am not sure what this means: x, y, and z, as well as
    left or right, are just names. The only important information between two
    coordinate systems is the transform linking them.


The sentence:

"The only important information between two coordinate systems is the transform
linking them."

has, in one form or another, often been repeated in NiPy meetings, but no one
bothers to define the terms in this sentence.  So, I have to ask what is your
definition of "transform" and "coordinate system"?  I have a precise definition,
and the names are part of it.

Let's go through that sentence. Mathematically, if a transform is a function,
then a transform knows its domain and its range so it knows the what the
coordinate systems are. So yes, with transform defined as "function", if I give
you a transform between two coordinate systems (mathematical spaces of some
kind) the only important information about it is itself.

The problem is that, for a 4x4 matrix T, the python function

transform_function = lambda v: np.dot(T, np.hstack([v,1])[:3]

has a "duck-type" domain that knows nothing about image acquisition and a range inferred by numpy that knows nothing about LPI or MNI152.  The string "coord_sys" in AffineImage is meant to imply that its domain and range say it should be interpreted in some way, but it is not explicit in AffineImage.

(Somewhere around here, I start veering off into documentation.... sorry).

To me, a "coordinate system" is a basis for a vector space (sometimes you might
want transforms between integers but ignore them for now).  It's not even a
description of an affine subspace of a vector space, (see e.g.
http://en.wikipedia.org/wiki/Affine_transformation).  To describe such an affine
subspace, "coordinate system" would need one more piece of information, the
"constant" or "displacement" vector of the affine subspace.

Because it's a basis, each element in the basis can be identified by a name, so
the transform depends on the names because that's how I determine a "coordinate
system" and I need "coordinate systems" because they are what the domain and
range of my "transform" are going to be. For instance, this describes the range
"coordinate system" of a "transform" whose output is in LPI coordinates:

"x" = a unit vector of length 1mm pointing in the Left to Right direction
"y" = a unit vector of length 1mm pointing in the Posterior to Anterior direction
"z" = a unit vector of length 1mm pointing in the Inferior to Superior direction

OK, so that's my definition of "coordinate system" and the names are an
important part of it.

Now for the "transform" which I will restrict to be "affine transform". To me,
this is an affine function or transformation between two vector spaces (we're
not even considering affine transformations between affine spaces). I bring up
the distinction because generally affine transforms act on affine spaces rather
than vector spaces. A vector space is an affine subspace of itself with
"displacement" vector given by its origin, hence it is an affine space and so we
can define affine functions on vector spaces.

Because it is an affine function, the mathematical image of the domain under
this function is an affine subspace of its range (which is a vector space). The
"displacement" vector of this affine subspace is represented by the floats in b
where A,b = to_matvec(T) (once I have specified a basis for the range of this
function).

Since my "affine transform" is a function between two vector spaces, it should
have a domain that is a vector space, as well. For the "affine transform"
associated with an Image, this domain vector space has coordinates that can be
interpreted as array coordinates, or coordinates in a "data cube". Depending on
the acquisition parameters, these coordinates might have names like "phase",
"freq", "slice".

Now, I can encode all this information in a tuple: (T=a 4x4 matrix of floats
with bottom row [0,0,0,1], ('phase', 'freq', "slice"), ('x','y','z'))

>>> from nipy.core.api import CoordinateSystem
>>> acquisition = ('phase', 'freq', 'slice')
>>> xyz_world = ('x','y','z')
>>> T = np.array([[2,0,0,-91.095],[0,2,0,-129.51],[0,0,2,-73.25],[0,0,0,1]])
>>> AffineTransform(CoordinateSystem(acquisition), CoordinateSystem(xyz_world), T)
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('phase', 'freq', 'slice'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64),
   affine=array([[   2.   ,    0.   ,    0.   ,  -91.095],
                 [   0.   ,    2.   ,    0.   , -129.51 ],
                 [   0.   ,    0.   ,    2.   ,  -73.25 ],
                 [   0.   ,    0.   ,    0.   ,    1.   ]])
)

The float64 appearing above is a way of specifying that the "coordinate systems"
are vector spaces over the real numbers, rather than, say the complex numbers.
It is specified as an optional argument to CoordinateSystem.

Compare this to the way a MINC file is described::

    jtaylo@ubuntu:~$ mincinfo data.mnc
    file: data.mnc
    image: signed__ short -32768 to 32767
    image dimensions: zspace yspace xspace
        dimension name         length         step        start
        --------------         ------         ----        -----
        zspace                     84            2       -73.25
        yspace                    114            2      -129.51
        xspace                     92            2      -91.095
    jtaylo@ubuntu:~$
    jtaylo@ubuntu:~$ mincheader data.mnc
    netcdf data {
    dimensions:
        zspace = 84 ;
        yspace = 114 ;
        xspace = 92 ;
    variables:
        double zspace ;
            zspace:varid = "MINC standard variable" ;
            zspace:vartype = "dimension____" ;
            zspace:version = "MINC Version    1.0" ;
            zspace:comments = "Z increases from patient inferior to superior" ;
            zspace:spacing = "regular__" ;
            zspace:alignment = "centre" ;
            zspace:step = 2. ;
            zspace:start = -73.25 ;
            zspace:units = "mm" ;
        double yspace ;
            yspace:varid = "MINC standard variable" ;
            yspace:vartype = "dimension____" ;
            yspace:version = "MINC Version    1.0" ;
            yspace:comments = "Y increases from patient posterior to anterior" ;
            yspace:spacing = "regular__" ;
            yspace:alignment = "centre" ;
            yspace:step = 2. ;
            yspace:start = -129.509994506836 ;
            yspace:units = "mm" ;
        double xspace ;
            xspace:varid = "MINC standard variable" ;
            xspace:vartype = "dimension____" ;
            xspace:version = "MINC Version    1.0" ;
            xspace:comments = "X increases from patient left to right" ;
            xspace:spacing = "regular__" ;
            xspace:alignment = "centre" ;
            xspace:step = 2. ;
            xspace:start = -91.0950012207031 ;
            xspace:units = "mm" ;
        short image(zspace, yspace, xspace) ;
            image:parent = "rootvariable" ;
            image:varid = "MINC standard variable" ;
            image:vartype = "group________" ;
            image:version = "MINC Version    1.0" ;
            image:complete = "true_" ;
            image:signtype = "signed__" ;
            image:valid_range = -32768., 32767. ;
            image:image-min = "--->image-min" ;
            image:image-max = "--->image-max" ;
        int rootvariable ;
            rootvariable:varid = "MINC standard variable" ;
            rootvariable:vartype = "group________" ;
            rootvariable:version = "MINC Version    1.0" ;
            rootvariable:parent = "" ;
            rootvariable:children = "image" ;
        double image-min ;
            image-min:varid = "MINC standard variable" ;
            image-min:vartype = "var_attribute" ;
            image-min:version = "MINC Version    1.0" ;
            image-min:_FillValue = 0. ;
            image-min:parent = "image" ;
        double image-max ;
            image-max:varid = "MINC standard variable" ;
            image-max:vartype = "var_attribute" ;
            image-max:version = "MINC Version    1.0" ;
            image-max:_FillValue = 1. ;
            image-max:parent = "image" ;
    data:

    zspace = 0 ;

    yspace = 0 ;

    xspace = 0 ;

    rootvariable = _ ;

    image-min = -50 ;

    image-max = 50 ;
    }

I like the MINC description, but the one thing missing in this file is the
ability to specify ('phase', 'freq', 'slice').  It may be possible to add it but
I'm not sure, it certainly can be added by adding a string to the header.  It
also mixes the definition of the basis with the affine transformation (look at
the output of mincheader which says that yspace has step 2). The NIFTI-1
standard allows limited possibilities to specify ('phase', 'freq', 'slice') this
with its dim_info byte but there are pulse sequences for which these names are
not appropriate.

One might ask: why bother making a "coordinate system" for the voxels. Well,
this is part of my definition of "affine transform".  More importantly, it
separates the notion of world axes ('x','y','z') and voxel indices
('i','j','k'). There is at least one use case, slice timing, a key step in the
fMRI pipeline, where we need to know which spatial axis is slice. One solution
would be to just add an attribute to AffineImage called "slice_axis" but then,
as Gael says, the possibilites for axis names are infinite, what if we want an
attribute for "group_axis"? AffineTransform provides an easy way to specify an
axis as "slice":

>>> unknown_acquisition = ('i','j','k')
>>> A = AffineTransform(CoordinateSystem(unknown_acquisition),
...                     CoordinateSystem(xyz_world), T)

After some deliberation, we find out that the third axis is slice...

>>> A.renamed_domain({'k':'slice'})
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'slice'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64),
   affine=array([[   2.   ,    0.   ,    0.   ,  -91.095],
                 [   0.   ,    2.   ,    0.   , -129.51 ],
                 [   0.   ,    0.   ,    2.   ,  -73.25 ],
                 [   0.   ,    0.   ,    0.   ,    1.   ]])
)

Or, working with an LPIImage rather than an AffineTransform

>>> from nipy.core.api import LPIImage
>>> data = np.random.standard_normal((92,114,84))
>>> im = LPIImage(data, A.affine, unknown_acquisition)
>>> im_slice_3rd = im.renamed_axes(k='slice')
>>> im_slice_3rd.lpi_transform
LPITransform(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'slice'), name='voxel', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPI', coord_dtype=float64),
   affine=array([[   2.   ,    0.   ,    0.   ,  -91.095],
                 [   0.   ,    2.   ,    0.   , -129.51 ],
                 [   0.   ,    0.   ,    2.   ,  -73.25 ],
                 [   0.   ,    0.   ,    0.   ,    1.   ]])
)

Note that A does not have 'voxel' or 'world-LPI' in it, but the lpi_transform
attribute of im does. The ('x','y','z') paired with ('world-LPI') is interpreted
to mean: "x is left-> right", "y is posterior-> anterior", "z is inferior to
superior", and the first number output from the python function
transform_function above is "x", the second is "y", the third is "z".

Another question one might ask is: why bother allowing non-4x4 affine matrices
like:

>>> AffineTransform.from_params('ij', 'xyz', np.array([[2,3,1,0],[3,4,5,0],[7,9,3,1]]).T)
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'j'), name='domain', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='range', coord_dtype=float64),
   affine=array([[ 2.,  3.,  7.],
                 [ 3.,  4.,  9.],
                 [ 1.,  5.,  3.],
                 [ 0.,  0.,  1.]])
)

For one, it allows very clear specification of a 2-dimensional plane (i.e. a
2-dimensional affine subspace of some vector spce) called P, in, say, the  LPI
"coordinate system". Let's say we want the plane in LPI-world corresponding to
"j=30" for im above. (I guess that's coronal?)

>>> # make an affine transform that maps (i,k) -> (i,30,k)
>>> j30 = AffineTransform(CoordinateSystem('ik'), CoordinateSystem('ijk'), np.array([[1,0,0],[0,0,30],[0,1,0],[0,0,1]]))
>>> j30
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'k'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
   affine=array([[  1.,   0.,   0.],
                 [  0.,   0.,  30.],
                 [  0.,   1.,   0.],
                 [  0.,   0.,   1.]])
)
>>> # it's dtype is np.float since we didn't specify np.int in constructing the CoordinateSystems

>>> j30_to_LPI = compose(im.lpi_transform, j30)
>>> j30_to_LPI
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'k'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPI', coord_dtype=float64),
   affine=array([[  2.   ,   0.   , -91.095],
                 [  0.   ,   0.   , -69.51 ],
                 [  0.   ,   2.   , -73.25 ],
                 [  0.   ,   0.   ,   1.   ]])
)

This could be used to resample any LPIImage on the coronal plane y=-69.51 with
voxels of size 2mmx2mm starting at x=-91.095 and z=-73.25. Of course, this
doesn't seem like a very natural slice. The module
:mod:`nipy.core.reference.slices` has some convenience functions for specifying
slices

>>> x_spec = ([-92,92], 93) # voxels of size 2 in x, starting at -92, ending at 92
>>> z_spec = ([-70,100], 86) # voxels of size 2 in z, starting at -70, ending at 100
>>> y70 = yslice(70, x_spec, z_spec, 'world-LPI')
>>> y70
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i_x', 'i_z'), name='slice', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPI', coord_dtype=float64),
   affine=array([[  2.,   0., -92.],
                 [  0.,   0.,  70.],
                 [  0.,   2., -70.],
                 [  0.,   0.,   1.]])
)

>>> bounding_box(y70, (x_spec[1], z_spec[1]))
    ([-92.0, 92.0], [70.0, 70.0], [-70.0, 100.0])

Maybe these aren't things that "normal human beings" (to steal a quote from
Gael) can use, but they're explicit and they are tied to precise mathematical
objects.

Email #2
---------

I apologize again for the long emails, but I'm glad we. as a group, are having
this discussion electronically. Usually, our discussions of CoordinateMap begin
with Matthew standing in front of a white board with a marker and asking a
newcomer,

"Are you familiar with the notion of a transformation, say, from voxel to world?"

:)

Where they go after that really depends on the kind of day everyone's having...

:)

These last two emails also have the advantage that most of them can go right in
to doc/users/coordinate_map.rst.

    I agree with Gael that LPIImage is an obscure name.

OK. I already know that people often don't agree with names I choose, just ask
Matthew. :)

I just wanted to choose a name that is as explicit as possible. Since I'm
neither a neuroscientist nor an MRI physicist but a statistician, I have no idea
what it really means. I found it mentioned in this link below and John Ollinger
mentioned LPI in another email thread

http://afni.nimh.nih.gov/afni/community/board/read.php?f=1&i=9140&t=9140

I was suggesting we use a well-established term, apparently LPI is not
well-established. :)

Does LPS mean (left, posterior, superior)?  Doesn't that suggest that LPI means
(left, posterior, inferior) and RAI means (right, anterior, inferior)?  If so,
then good, now I know what LPI means and I'm not a neuroscientist or an MRI
physicist, :)

We can call the images RASImages, or at least let's call their AffineTransform
RASTransforms, or we could have NeuroImages that can only have RASTransforms or
LPSTransforms, NeuroTransform that have a property and NeuroImage raises an
exception like this:

@property
def world(self):
   return self.affine_transform.function_range

if self.world.name not in ['world-RAS', 'world-LPS'] or self.world.coord_names != ('x', 'y', 'z'):
    raise ValueError("the output space must be named one of ['world-RAS','world-LPS'] and the axes must be ('x', 'y', 'z')")

_doc['world'] = "World space, one of ['world-RAS', 'world-LPS']. If it is 'world-LPS', then x increases from patient's left to right, y increases posterior to anterior, z increases superior to inferior. If it is 'world-RAS' then x increases patient's right to left, y increases posterior to anterior, z increases superior to inferior."

I completely advocate any responsibility for deciding which acronym to choose,
someone who can use rope can just change every lpi/LPI to ras/RAS I just want it
explicit.  I also want some version of these phrases "x increases from patient's
right to left", "y increases from posterior to anterior", "z increases from
superior to inferior" somewhere in a docstring for RAS/LPSTransform (see why I
feel that "increasing vs. decreasing" is important below).

I want the name and its docstring to scream at you what it represents so there
is no discussion like on the AFNI list where users are not sure which output of
which program (in AFNI) should be flipped (see the other emails in the thread).
It should be a subclass of AffineTransform because it has restrictions: namely,
its range is 'xyz'  and "xy" can be interpreted in of two ways either RAS or
LPS). You can represent any other version of RAS/LPS or (whatever colour your
bike shed is, :)) with the same class, it just may have negative values on the
diagonal. If it has some rotation applied, then it becomes pretty hard (at least
for me) to decide if it's RAS or LPS from the 4x4 matrix of floats. I can't even
tell you now when I look at the FIAC data which way left and right go unless I
ask Matthew.

    For background, you may want to look at what Gordon Kindlmann did for
    nrrd format where you can declare the space in which your orientation
    information and other transforms should be interpreted:

    http://teem.sourceforge.net/nrrd/format.html#space

    Or, if that's too flexible for you, you could adopt a standard space. 


    ITK chose LPS to match DICOM. 

    For slicer, like nifti, we chose RAS

It may be that there is well-established convention for this, but then why does
ITK say DICOM=LPS and AFNI say DICOM=RAI?  At least MINC is explicit. I favor
making it as precise as MINC does. 

That AFNI discussion I pointed to uses the pairing RAI/DICOM and LPI/SPM.  This
discrepancy suggests there's some disagreement between using the letters to name
the system and whether they mean increasing or decreasing. My guess is that
LPI=RAS based on ITK/AFNI's identifications of LPS=DICOM=RAI. But I can't tell
if the acronym LPI means "x is increasing L to R, y increasing from P to A, z in
increasing from I to S" which would be equivalent to RAS meaning "x decreasing
from R to L, y decreasing from A to P, z is decreasing from S to I". That is, I
can't tell from the acronyms which of LPI or RAS is using "increasing" and which
is "decreasing", i.e. they could have flipped everything so that LPI means "x is
decreasing L to R, y is decreasing P to A, z is decreasing I to S" and RAS means
"x is increasing R to L, y is increasing A to P, z is increasing S to I".

To add more confusion to the mix, the acronym doesn't say if it is the patient's
left to right or the technician looking at him, :) For this, I'm sure there's a
standard answer, and it's likely the patient, but heck, I'm just a statistician
so I don't know the answer.


    (every volume has an ijkToRAS affine transform).  We convert to/from LPS
    when calling ITK code, e.g., for I/O.

How much clearer can you express "ijkToRAS" or "convert to/from LPS" than
something like this:

>>> T = np.array([[2,0,0,-91.095],[0,2,0,-129.51],[0,0,2,-73.25],[0,0,0,1]])
>>> ijk = CoordinateSystem('ijk', 'voxel')
>>> RAS = CoordinateSystem('xyz', 'world-RAS')
>>> ijk_to_RAS = AffineTransform(ijk, RAS, T)
>>> ijk_to_RAS
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('R', 'A', 'S'), name='', coord_dtype=float64),
   affine=array([[   2.   ,    0.   ,    0.   ,  -91.095],
                 [   0.   ,    2.   ,    0.   , -129.51 ],
                 [   0.   ,    0.   ,    2.   ,  -73.25 ],
                 [   0.   ,    0.   ,    0.   ,    1.   ]])
)

>>> LPS = CoordinateSystem('xyz', 'world-LPS')
>>> RAS_to_LPS = AffineTransform(RAS, LPS, np.diag([-1,-1,1,1])) 
>>> ijk_to_LPS = compose(RAS_to_LPS, ijk_to_RAS)
>>> RAS_to_LPS
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-RAS', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPS', coord_dtype=float64),
   affine=array([[-1.,  0.,  0.,  0.],
                 [ 0., -1.,  0.,  0.],
                 [ 0.,  0.,  1.,  0.],
                 [ 0.,  0.,  0.,  1.]])
)
>>> ijk_to_LPS
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='voxel', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPS', coord_dtype=float64),
   affine=array([[  -2.   ,    0.   ,    0.   ,   91.095],
                 [   0.   ,   -2.   ,    0.   ,  129.51 ],
                 [   0.   ,    0.   ,    2.   ,  -73.25 ],
                 [   0.   ,    0.   ,    0.   ,    1.   ]])
)

Of course, we shouldn't rely on the names ijk_to_RAS to know that it is an
ijk_to_RAS transform, that's why they're in the AffineTransform. I don't think
any one wants an attribute named "ijk_to_RAS" for AffineImage/Image/LPIImage.

The other problem that LPI/RAI/AffineTransform addresses is that someday you
might want to transpose the data in your array and still have what you would
call an "image". AffineImage allows this explicitly because there is no
identifier for the domain of the AffineTransform (the attribute name "coord_sys"
implies that it refers to either the domain or the range but not both). (Even
those who share the sentiment that "everything that is important about the
linking between two coordinate systems is contained in the transform"
acknowledge there are two coordinate systems :))

Once you've transposed the array, say

>>> newdata = data.transpose([2,0,1])

You shouldn't use something called "ijk_to_RAS" or "ijk_to_LPS" transform.
Rather, you should use a "kij_to_RAS" or "kij_to_LPS" transform.

>>> kji = CoordinateSystem('kji')
>>> ijk_to_kij = AffineTransform(ijk, kij, np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]))
>>> import sympy
>>> # Check that it does the right permutation
>>> i, j, k = [sympy.Symbol(s) for s in 'ijk']
>>> ijk_to_kij([i,j,k])
array([k, i, j], dtype=object)
>>> # Yup, now let's try to make a kij_to_RAS transform
>>> # At first guess, we might try
>>> kij_to_RAS = compose(ijk_to_RAS,ijk_to_kij)
------------------------------------------------------------
Traceback (most recent call last):
  File "<ipython console>", line 1, in <module>
  File "reference/coordinate_map.py", line 1090, in compose
    return _compose_affines(*cmaps)
  File "reference/coordinate_map.py", line 1417, in _compose_affines
    raise ValueError("domains and ranges don't match up correctly")
ValueError: domains and ranges don't match up correctly

>>> # but we have a problem, we've asked for a composition that doesn't make sense

If you're good with permutation matrices, you wouldn't have to call "compose"
above and you can just do matrix multiplication.  But here the name of the
function tells you that yes, you should do the inverse: "ijk_to_kij" says that
the range are "kij" values, but to get a "transform" for your data in "kij" it
should have a domain that is "kij" so it should be

The call to compose raised an exception because it saw you were trying to
compose a function with domain="ijk" and range="kji" with a function (on its
left) having domain="ijk" and range "kji". This composition just doesn't make
sense so it raises an exception.

>>> kij_to_ijk = ijk_to_kij.inverse()
>>> kij_to_RAS = compose(ijk_to_RAS,kij_to_ijk)
>>> kij_to_RAS
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-RAS', coord_dtype=float64),
   affine=array([[   0.   ,    2.   ,    0.   ,  -91.095],
                 [   0.   ,    0.   ,    2.   , -129.51 ],
                 [   2.   ,    0.   ,    0.   ,  -73.25 ],
                 [   0.   ,    0.   ,    0.   ,    1.   ]])
)


>>> ijk_to_RAS([i,j,k])
array([-91.095 + 2.0*i, -129.51 + 2.0*j, -73.25 + 2.0*k], dtype=object)
>>> kij_to_RAS([k,i,j])
array([-91.095 + 2.0*i, -129.51 + 2.0*j, -73.25 + 2.0*k], dtype=object)
>>>
>>> another_kij_to_RAS([k,i,j])
array([-91.095 + 2.0*i, -129.51 + 2.0*j, -73.25 + 2.0*k], dtype=object)

We also shouldn't have to rely on the names of the AffineTransforms, i.e.
ijk_to_RAS,  to remember what's what (in typing this example, I mixed up kij and
kji many times). The three objects ijk_to_RAS, kij_to_RAS and another_kij_to_RAS
all represent the same "affine transform", as evidenced by their output above.
There are lots of representations of the same "affine transform":
(6=permutations of i,j,k)*(6=permutations of x,y,z)=36 matrices for one "affine
transform".

If we throw in ambiguity about the sign in front of the output, there are
36*(8=2^3 possible flips of the x,y,z)=288 matrices possible but there are only
really 8 different "affine transforms". If you force the order of the range to
be "xyz" then there are 6*8=48 different matrices possible, again only
specifying 8 different "affine transforms". For AffineImage, if we were to allow
both "LPS" and "RAS" this means two flips are allowed, namely either
"LPS"=[-1,-1,1] or "RAS"=[1,1,1], so there are 6*2=12 possible matrices to
represent 2 different "affine transforms".

Here's another example that uses sympy to show what's going on in the 4x4 matrix
as you reorder the 'ijk' and the 'RAS'. (Note that this code won't work in
general because I had temporarily disabled a check in CoordinateSystem that
enforced the dtype of the array to be a builtin scalar dtype for sanity's sake).
To me, each of A, A_kij and A_kij_yzx below represent the same "transform"
because if I substitue i=30, j=40, k=50 and I know the order of the 'xyz' in the
output then they will all give me the same answer.

    >>> ijk = CoordinateSystem('ijk', coord_dtype=np.array(sympy.Symbol('x')).dtype)
    >>> xyz = CoordinateSystem('xyz', coord_dtype=np.array(sympy.Symbol('x')).dtype)
    >>> x_start, y_start, z_start = [sympy.Symbol(s) for s in ['x_start', 'y_start', 'z_start']]
    >>> x_step, y_step, z_step = [sympy.Symbol(s) for s in ['x_step', 'y_step', 'z_step']]
    >>> i, j, k = [sympy.Symbol(s) for s in 'ijk']
    >>> T = np.array([[x_step,0,0,x_start],[0,y_step,0,y_start],[0,0,z_step,z_start],[0,0,0,1]])
    >>> T
    array([[x_step, 0, 0, x_start],
           [0, y_step, 0, y_start],
           [0, 0, z_step, z_start],
           [0, 0, 0, 1]], dtype=object)
    >>> A = AffineTransform(ijk, xyz, T)
    >>> A
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=object),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=object),
       affine=array([[x_step, 0, 0, x_start],
                     [0, y_step, 0, y_start],
                     [0, 0, z_step, z_start],
                     [0, 0, 0, 1]], dtype=object)
    )
    >>> A([i,j,k])
    array([x_start + i*x_step, y_start + j*y_step, z_start + k*z_step], dtype=object)
    >>> # this is another
    >>> A_kij = A.reordered_domain('kij')

    >>> A_kij
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='', coord_dtype=object),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=object),
       affine=array([[0, x_step, 0, x_start],
                     [0, 0, y_step, y_start],
                     [z_step, 0, 0, z_start],
                     [0.0, 0.0, 0.0, 1.0]], dtype=object)
    )
    >>>
    >>> A_kij([k,i,j])
    array([x_start + i*x_step, y_start + j*y_step, z_start + k*z_step], dtype=object)
                                                                                    >>> # let's look at another reordering
    >>> A_kij_yzx = A_kij.reordered_range('yzx')
    >>> A_kij_yzx
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='', coord_dtype=object),
       function_range=CoordinateSystem(coord_names=('y', 'z', 'x'), name='', coord_dtype=object),
       affine=array([[0, 0, y_step, y_start],
                     [z_step, 0, 0, z_start],
                     [0, x_step, 0, x_start],
                     [0, 0, 0, 1.00000000000000]], dtype=object)
    )
    >>> A_kij_yzx([k,i,j])
    array([y_start + j*y_step, z_start + k*z_step, x_start + i*x_step], dtype=object)
    >>>

>>> A_kij
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('k', 'i', 'j'), name='', coord_dtype=object),
   function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=object),
   affine=array([[0, x_step, 0, x_start],
                 [0, 0, y_step, y_start],
                 [z_step, 0, 0, z_start],
                 [0.0, 0.0, 0.0, 1.0]], dtype=object)
)

>>> equivalent(A_kij, A)
True
>>> equivalent(A_kij, A_kij_yzx)
True

