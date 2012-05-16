.. _coordinate_map:

#############################
 Basics of the Coordinate Map
#############################

When you load an image it will have an associated Coordinate Map

**Coordinate Map**

    The Coordinate Map contains information defining the input (domain) and
    output (range) Coordinate Systems of the image, and the mapping between the
    two Coordinate systems.

The *input* or *domain* in an image are voxel coordinates in the image array.
The *output* or *range* are the millimetre coordinates in some space, that
correspond to the input (voxel) coordinates.

>>> import nipy

Get a filename for an example file:

>>> from nipy.testing import anatfile

Get the coordinate map for the image:

>>> anat_img = nipy.load_image(anatfile)
>>> coordmap = anat_img.coordmap

For more on Coordinate Systems and thier properties
:mod:`nipy.core.reference.coordinate_system`

You can inspect a coordinate map::

>>> coordmap.function_domain.coord_names
>>> ('i', 'j', 'k')

>>> coordmap.function_range.coord_names

>>> coordmap.function_domain.name
'input'
>>> coordmap.function_range.name
'output'

A Coordinate Map has a mapping from the *input* Coordinate System to the
*output* Coordinate System

Here we can see we have a voxel to millimeter mapping from the voxel
space (i,j,k) to the millimeter space (x,y,z)

We can also get the name of the respective Coordinate Systems that our
Coordinate Map maps between.

A Coordinate Map is two Coordinate Systems with a mapping between
them.  Formally the mapping is a function that takes points from the
input Coordinate System and returns points from the output Coordinate
System.  This is the same as saying that the mapping takes points in the mapping
function *domain* and transforms them to points in the mapping function *range*.

Often this is simple as applying an Affine transform. In that case the
Coordinate System may well have an affine property which returns the
affine matrix corresponding to the transform.

>>> coordmap.affine
array([[ -2.,   0.,   0.,  32.],
       [  0.,   2.,   0., -40.],
       [  0.,   0.,   2., -16.],
       [  0.,   0.,   0.,   1.]])

If you call the Coordinate Map you will apply the mapping function
between the two Coordinate Systems. In this case from (i,j,k) to (x,y,z):

>>> coordmap([1,2,3])
array([ 30., -36., -10.])

It can also be used to  get the inverse mapping, or in this example from (x,y,z)
back to (i,j,k):

>>> coordmap.inverse()([30.,-36.,-10.])
array([ 1.,  2.,  3.])

We can see how this works if we just apply the affine
ourselves using dot product.

.. Note::

    Notice the affine is using homogeneous coordinates so we need to add a 1 to
    our input. (And note how  a direct call to the coordinate map does this work
    for you)

>>> coordmap.affine
array([[ -2.,   0.,   0.,  32.],
       [  0.,   2.,   0., -40.],
       [  0.,   0.,   2., -16.],
       [  0.,   0.,   0.,   1.]])

>>> import numpy as np
>>> np.dot(coordmap.affine, np.transpose([1,2,3,1]))
array([ 30., -36., -10.,   1.])

.. Note::

   The answer is the same as above (except for the added 1)

***************************************************
Use of the Coordinate Map for spatial normalization
***************************************************

The Coordinate Map can be used to describe the transformations needed to perform
spatial normalization. Suppose we have an anatomical Image from one subject
*subject_img* and we want to create an Image in a standard space like Tailarach
space. An affine registration algorithm will produce a 4-by-4 matrix
representing the affine transformation, *T*, that takes a point in the subject's
coordinates *subject_world* to a point in Tailarach space *tailarach_world*. The
subject's Image has its own Coordinate Map, *subject_cmap* and there is a
Coordinate Map for Tailarach space which we will call *tailarach_cmap*.

Having found the transformation matrix *T*, the next step in spatial
normalization is usually to resample the array of *subject_img* so that it has
the same shape as some atlas *atlas_img*. Note that because it is an atlas
Image, *tailarach_camp=atlas_img.coordmap*.

A resampling algorithm uses an interpolator which needs to know
which voxel of *subject_img* corresponds to which voxel of *atlas_img*.
This is therefore a function from *atlas_voxel* to *subject_voxel*.

This function, paired with the information that it is a map from atlas-voxel to
subject-voxel is another example of a Coordinate Map. The code to do this might
look something like the following:

>>> from nipy.testing import anatfile, funcfile
>>> from nipy.algorithms.registration import HistogramRegistration
>>> from nipy.algorithms.kernel_smooth import LinearFilter

We'll make a smoothed version of the anatomical example image, and pretend it's
the template

>>> smoother = LinearFilter(anat_img.coordmap, anat_img.shape)
>>> atlas_im = smoother.smooth(anat_img)
>>> subject_im = anat_img

We do an affine registration between the two.

>>> reggie = HistogramRegistration(subject_im, atlas_im)
>>> aff = reggie.optimize('affine').as_affine() #doctest: +ELLIPSIS
Initial guess...
...

Now we make a coordmap with this transformation

>>> from nipy.core.api import AffineTransform
>>> subject_cmap = subject_im.coordmap
>>> talairach_cmap = atlas_im.coordmap
>>> subject_world_to_talairach_world = AffineTransform(
...                                       subject_cmap.function_range,
...                                       talairach_cmap.function_range,
...                                       aff)
...

We resample the 'subject' image to the 'atlas image

>>> from nipy.algorithms.resample import resample
>>> normalized_subject_im = resample(subject_im, talairach_cmap,
...                                  subject_world_to_talairach_world,
...                                  atlas_im.shape)
>>> normalized_subject_im.shape == atlas_im.shape
True
>>> normalized_subject_im.coordmap == atlas_im.coordmap
True
>>> np.all(normalized_subject_im.affine == atlas_im.affine)
True

**********************************************
Mathematical formulation of the Coordinate Map
**********************************************

Using the *CoordinateMap* can be a little hard to get used to.  For some users,
a mathematical description, free of any python syntax and code design and
snippets may be helpful. After following through this description, the code
design and usage should hopefully be clearer.

We return to the normalization example and try to write it out mathematically.
Conceptually, to do normalization, we need to be able to answer each of these
three questions:

1. *Voxel-to-world (subject)* Given the subjects' anatomical image read off the
   scanner: which physical location, expressed in :math:`(x_s,y_s,z_s)`
   coordinates (:math:`s` for subject), corresponds to the voxel of data
   :math:`(i_s,j_s,k_s)`?  This question is answered by *subject_im.coordmap*.
   The actual function that computes this, i.e that takes 3 floats and returns 3
   floats, is *subject_im.coordmap.mapping*.
2. *World-to-world (subject to Tailarach)* Given a location
   :math:`(x_s,y_s,z_s)` in an anatomical image of the subject, where does it
   lie in the Tailarach coordinates :math:`(x_a,y_a, z_a)`? This is answered by
   the matrix *T* and knowing that *T* maps a point in the subject's world to
   Tailarach world. Hence, this question is answered by
   *subject_world_to_tailarach_world* above.
3. *Voxel-to-world (Tailarach)* Since we want to produce a resampled Image that
   has the same shape and coordinate information as *atlas_im*, we need to know
   what location in Tailarach space, :math:`(x_a,y_a,z_a)` (:math:`a` for atlas)
   corresponds to the voxel :math:`(i_a,j_a,k_a)`. This question is answered by
   *tailarach_cmap*.

Each of these three questions are answered by, in code, what we called a class
called *CoordinateMap*.  Mathematically, let's define a *mapping* as a tuple
:math:`(D,R,f)` where :math:`D` is the *domain*, :math:`R` is the *range* and
:math:`f:D\rightarrow R` is a function. It may seem redundant to pair
:math:`(D,R)` with :math:`f` because a function must surely know its domain and
hence, implicitly, its range.  However, we will see that when it comes time to
implement the notion of *mapping*, the tuple we do use to construct
*CoordinateMap* is almost, but not quite :math:`(D,R,f)` and, in the tuple we
use, :math:`D` and :math:`R` are not reduntant.

Since these mappings are going to be used and called with modules like
:mod:`numpy`, we should restrict our definition a little bit. We assume the
following:

1. :math:`D` is isomorphic to one of :math:`\mathbb{Z}^n, \mathbb{R}^n,
   \mathbb{C}^n` for some :math:`n`. This isomorphism is determined by a basis
   :math:`[u_1,\dots,u_n]` of :math:`D` which maps :math:`u_i` to :math:`e_i`
   the canonical i-th coordinate vector of whichever of :math:`\mathbb{Z}^n,
   \mathbb{R}^n, \mathbb{C}^n`. This isomorphism is denoted by :math:`I_D`.
   Strictly speaking, if :math:`D` is isomorphic to :math:`\mathbb{Z}^n` then
   the term basis is possibly misleading because :math:`D` because it is not a
   vector space, but it is a group so we might call the basis a set of
   generators instead. In any case, the implication is that whatever properties
   the appropriate :math:`\mathbb{Z},\mathbb{R},\mathbb{C}`, so :math:`D` (and
   :math:`R`) has as well.
2. :math:`R` is similarly isomorphic to one of  :math:`\mathbb{Z}^m,
   \mathbb{R}^m, \mathbb{C}^m` for some :math:`m` with isomorphism :math:`I_R`
   and basis :math:`[v_1,\dots,v_m]`.

Above, and throughout, the brackets "[","]" represent things interpretable as
python lists, i.e. sequences.

These isomorphisms are just fancy ways of saying that the point
:math:`x=3,y=4,z=5` is represented by the 3 real numbers (3,4,5). In this case
the basis is :math:`[x,y,z]` and for any :math:`a,b,c \in \mathbb{R}`

.. math::

   I_D(a\cdot x + b \cdot y + c \cdot z) = a \cdot e_1 + b \cdot e_2 + c \cdot e_3

We might call the pairs :math:`([u_1,...,u_n], I_D), ([v_1,...,v_m], I_R)`
*coordinate systems*.  Actually, the bases in effect determine the maps
:math:`I_D,I_R` as long as we know which of
:math:`\mathbb{Z},\mathbb{R},\mathbb{C}` we are talking about so in effect,
:math:`([u_1,...,u_n], \mathbb{R})` could be called a *coordinate system*.  This
is how it is implemented in the code with :math:`[u_1, \dots, u_n]` being
replaced by a list of strings naming the basis vectors and :math:`\mathbb{R}`
replaced by a builtin :func:`numpy.dtype`.

In our normalization example, we therefore have 3 mappings:

1. *Voxel-to-world (subject)* In standard notation for functions, we can write

   .. math::

      (i_s,j_s,k_s) \overset{f}{\mapsto} (x_s,y_s,z_s).

   The domain is :math:`D=[i_s,j_s,k_s]`, the range is :math:`R=[x_s,y_s,z_s]`
   and the function is :math:`f:D \rightarrow R`.

2. *World-to-world (subject to Tailarach)* Again, we can write

   .. math::

      (x_s,y_s,z_s) \overset{g}{\mapsto} (x_a,y_a,z_a)

   The domain is :math:`D=[x_s,y_s,z_s]`, the range is :math:`R=[x_a,y_a,z_a]`
   and the function is :math:`g:D \rightarrow R`.

3. *Voxel-to-world (Tailarach)* Again, we can write

   .. math::

      (i_a,j_a,k_a) \overset{h}{\mapsto} (x_a,y_a, z_a).

   The domain is :math:`D=[i_a,j_a,k_a]`, the range is :math:`R=[x_a,y_a,z_a]`
   and the function is :math:`h:D \rightarrow R`.

Note that each of the functions :math:`f,g,h` can be, when we know the necessary
isomorphisms, thought of as functions from :math:`\mathbb{R}^3` to itself. In
fact, that is what we are doing when we write

   .. math::

      (i_a,j_a,k_a) \overset{h}{\mapsto} (x_a,y_a, z_a)

as a function that takes 3 numbers and gives 3 numbers.

Formally, these functions that take 3 numbers and return 3 numbers can be
written as :math:`\tilde{f}=I_R \circ f \circ I_D^{-1}`.  When this is
implemented in code, it is actually the functions :math:`\tilde{f}, \tilde{g},
\tilde{h}` we specify, rather then :math:`f,g,h`. The functions
:math:`\tilde{f}, \tilde{g}, \tilde{h}`  have domains and ranges that are just
:math:`\mathbb{R}^3`.  We therefore call a *coordinate map*  a tuple

.. math::

   ((u_D, \mathbb{R}), (u_R, \mathbb{R}), I_R \circ f \circ I_D^{-1})

where :math:`u_D, u_R` are bases for :math:`D,R`, respectively.  It is this
object that is implemented in code. There is a simple relationship between
*mappings* and *coordinate maps*

.. math::

   ((u_D, \mathbb{R}), (u_R, \mathbb{R}), \tilde{f}) \leftrightarrow (D, R, f=I_R^{-1} \circ \tilde{f} \circ I_D)

Because :math:`\tilde{f}, \tilde{g}, \tilde{h}` are just functions from
:math:`\mathbb{R}^3` to itself, they can all be composed with one another. But,
from our description of the functions above, we know that only certain
compositions make sense and others do not, such as :math:`g \circ h`.
Compositions that do make sense include

1. :math:`h^{-1} \circ g` which :math:`(i_a,j_a, k_a)` voxel corresponds to the
   point :math:`(x_s,y_s,z_s)`?
2. :math:`g \circ f` which :math:`(x_a,y_a,z_a)` corresponds to the voxel
   :math:`(i,j,k)`?

The composition that is used in the normalization example is :math:`w = f^{-1}
\circ g^{-1} \circ h` which is a function

.. math::

   (i_a, j_a, k_a) \overset{w}{\mapsto} (i_s, j_s, k_s)

This function, or more correctly its representation :math:`\tilde{w}` that takes
3 floats to 3 floats, is passed directly to
:func:`scipy.ndimage.map_coordinates`.

Manipulating mappings, coordinate systems and coordinate maps
=============================================================

In order to solve our normalization problem, we will definitely need to compose
functions. We may want to carry out other formal operations as well. Before
describing operations on mappings, we describe the operations you might want to
consider on coordinate systems.

Coordinate systems
------------------

1. *Reorder*: This is just a reordering of the basis, i.e.
   :math:`([u_1,u_2,u_3], \mathbb{R}) \mapsto ([u_2,u_3,u_1], \mathbb{R})`
2. *Product*: Topological product of the coordinate systems (with a small
   twist). Given two coordinate systems :math:`([u_1,u_2,u_3], \mathbb{R}),
   ([v_1, v_2], \mathbb{Z})` the product is represented as

   .. math::

      ([u_1,u_2,u_3], \mathbb{R}) \times ([v_1, v_2], \mathbb{Z})  \mapsto ([u_1,u_2,u_3,v_1,v_2], \mathbb{R})`. 

   Note that the resulting coordinate system is real valued whereas one of the
   input coordinate systems was integer valued. We can always embed
   :math:`\mathbb{Z}` into :math:`\mathbb{R}`.  If one of them is complex
   valued, the resulting coordinate system is complex valued. In the code, this
   is handled by attempting to find a safe builtin numpy.dtype for the two (or
   more) given coordinate systems.

Mappings
--------

1. *Inverse*: Given a mapping :math:`M=(D,R,f)` if the function :math:`f` is
   invertible, this is just the obvious :math:`M^{-1}=(R, D, f^{-1})`.
2. *Composition*: Given two mappings, :math:`M_f=(D_f, R_f, f)` and
   :math:`M_g=(D_g, R_g, g)` if :math:`D_f == R_g` then the composition is well
   defined and the composition of the mappings :math:`[M_f,M_g]` is just
   :math:`(D_g, R_f, f \circ g)`.
3. *Reorder domain / range*: Given a mapping :math:`M=(D=[i,j,k], R=[x,y,z], f)`
   you might want to specify that we've changed the domain by changing the
   ordering of its basis to :math:`[k,i,j]`. Call the new domain :math:`D'`.
   This is represented by the composition of the mappings :math:`[M, O]` where
   :math:`O=(D', D, I_D^{-1} \circ f_O \circ I_{D'})` and for  :math:`a,b,c \in
   \mathbb{R}`:

   .. math::

      f_O(a,b,c) = (b,c,a).

4. *Linearize*: Possibly less used, since we know that :math:`f` must map one of
   :math:`\mathbb{Z}^n, \mathbb{R}^n, \mathbb{C}^n` to one of
   :math:`\mathbb{Z}^m, \mathbb{R}^m, \mathbb{C}^m`, we might be able
   differentiate it at a point :math:`p \in D`, yielding its 1st order Taylor
   approximation

   .. math::

      f_p(d) = f(d) + Df_p(d-p)

   which is  an affine  function, thus
   creating an affine mapping :math:`(D, R, f_p)`. Affine functions
   are discussed in more detail below.

5. *Product*: Given two mappings :math:`M_1=(D_1,R_1,f_1), M_2=(D_2, R_2, f_2)`
   we define their product as the mapping :math:`(D_1 + D_2, R_1 + R_2, f_1
   \otimes f_2)` where

   .. math::

      (f_1 \otimes f_2)(d_1, d_2) = (f_1(d_1), f_2(d_2)).

   Above, we have taken the liberty of expressing the product of the coordinate
   systems, say, :math:`D_1=([u_1, \dots, u_n], \mathbb{R}), D_2=([v_1, \dots,
   v_m], \mathbb{C})` as a python addition of lists.

   The name *product* for this operation is not necessarily canonical. If the
   two coordinate systems are  vector spaces and the function is linear, then we
   might call this map the *direct sum* because its domain are direct sums of
   vector spaces. The term *product* here refers to the fact that the domain and
   range are true topological products.

Affine mappings
---------------

An *affine mapping* is one in which the function :math:`f:D \rightarrow R` is an
affine function. That is, it can be written as `f(d) = Ad + b` for :math:`d \in
D` for some :math:`n_R \times n_D` matrix :math:`A` with entries that are in one
of :math:`\mathbb{Z}, \mathbb{R}, \mathbb{C}`.

Strictly speaking, this is a little abuse of notation because :math:`d` is a
point in :math:`D` not a tuple of real (or integer or complex) numbers. The
matrix :math:`A` represents a linear transformation from :math:`D` to :math:`R`
in a particular choice of bases for :math:`D` and :math:`R`.

Let us revisit some of the operations on a mapping as applied to *affine
mappings* which we write as a tuple :math:`M=(D, R, T)` with :math:`T` the
representation of the :math:`(A,b)` in homogeneous coordinates.

1. *Inverse*: If :math:`T` is invertible, this is just the tuple
   :math:`M^{-1}=(R, D, T^{-1})`.

2. *Composition*: The composition of two affine mappings :math:`[(D_2, R_2,
   T_2), (D_1,R_1,T_1)]` is defined whenever :math:`R_1==D_2` and is the tuple
   :math:`(D_1, R_2, T_2 T_1)`.

3. *Reorder domain*: A reordering of the domain of an affine mapping
   :math:`M=(D, R, T)` can be represented by a :math:`(n_D+1) \times (n_D+1)`
   permutation matrix :math:`P` (in which the last coordinate is unchanged --
   remember we are in homogeneous coordinates). Hence a reordering of :math:`D`
   to :math:`D'` can be represented as :math:`(D', R, TP)`. Alternatively, it is
   the composition of the affine mappings :math:`[M,(\tilde{D}, D, P)]`.

4. *Reorder range*:  A reordering of the range can  be represented by a
   :math:`(n_R+1) \times (n_R+1)` permutation matrix :math:`\tilde{P}`.  Hence a
   reordering of :math:`R` to :math:`R'` can be represented as :math:`(D,
   \tilde{R}, \tilde{P}T)`. Alternatively, it is the composition of the affine
   mappings :math:`[(R, \tilde{R}, \tilde{P}), M]`.

5. *Linearize*: Because the mapping :math:`M=(D,R,T)` is already affine, this
   leaves it unchanged.

6. *Product*: Given two affine mappings :math:`M_1=(D_1,R_1,T_1)` and
   :math:`M_2=(D_2,R_2,T_2)` the product is the tuple

   .. math::

      \left(D_1+D_2,R_1+R_2,
        \begin{pmatrix}
        T_1 & 0 \\
        0 & T_2
        \end{pmatrix} \right).


3-dimensional affine mappings
-----------------------------

For an Image, by far the most common mappings associated to it are affine, and
these are usually maps from a real 3-dimensional domain to a real 3-dimensional
range. These can be represented by the ubiquitous :math:`4 \times 4` matrix (the
representation of the affine mapping in homogeneous coordinates), along with
choices for the axes, i.e. :math:`[i,j,k]` and the spatial coordinates, i.e.
:math:`[x,y,z]`.

We will revisit some of the operations on mappings  as applied specifically to
3-dimensional affine mappings which we write as a tuple :math:`A=(D, R, T)`
where :math:`T` is an invertible :math:`4 \times 4`  transformation matrix with
real entries.

1. *Inverse*: Because we have assumed that :math:`T` is invertible this is just  tuple :math:`(([x,y,z], \mathbb{R}), ([i,j,k], \mathbb{R}), T^{-1})`.

2. *Composition*: Given two 3-dimensional affine mappings :math:`M_1=(D_1,R_1,
   T_1), M_2=(D_2,R_2,T_2)` the composition of :math:`[M_2,M_1]` yields another
   3-dimensional affine mapping whenever :math:`R_1 == D_2`. That is, it yields
   :math:`(D_1, R_2, T_2T_1)`.

3. *Reorder domain* A reordering of the domain can be represented by a :math:`4
   \times 4` permutation matrix :math:`P` (with its last coordinate not
   changing). Hence the reordering of :math:`D=([i,j,k], \mathbb{R})` to
   :math:`([k,i,j], \mathbb{R})` can be represented as :math:`(([k,i,j],
   \mathbb{R}), R, TP)`. 

4. *Reorder range*: A reordering of the range can also be represented by a
   :math:`4 \times 4` permutation matrix :math:`\tilde{P}` (with its last
   coordinate not changing). Hence the reordering of :math:`R=([x,y,z],
   \mathbb{R})` to :math:`([z,x,y], \mathbb{R})` can be represented as
   :math:`(D, ([z,x,y], \mathbb{R}), \tilde{P}, T)`.

5. *Linearize*: Just as for a general affine mapping, this does nothing.

6. *Product*: Because we are dealing with only 3-dimensional mappings here, it
   is impossible to use the product because that would give a mapping between
   spaces of dimension higher than 3.

Coordinate maps
---------------

As noted above *coordinate maps* are equivalent to *mappings* through the
bijection

.. math::

   ((u_D, \mathbb{R}), (u_R, \mathbb{R}), \tilde{f}) \leftrightarrow (D, R, I_R^{-1} \circ \tilde{f} \circ I_D)

So, any manipulations on *mappings*, *affine mappings* or *3-dimensional affine
mappings* can be carried out on *coordinate maps*, *affine coordinate maps* or
*3-dimensional affine coordinate maps*.

Implementation
==============

Going from this mathematical description to code is fairly straightforward.

1. A *coordinate system* is implemented by the class *CoordinateSystem* in the
   module :mod:`nipy.core.reference.coordinate_system`. Its constructor takes a
   list of names, naming the basis vectors of the *coordinate system* and an
   optional built-in numpy scalar dtype such as np.float32.  It has no
   interesting methods of any kind. But there is a module level function
   *product* which implements the notion of the product of *coordinate systems*.

2. A *coordinate map* is implemented by the class *CoordinateMap* in the module
   :mod:`nipy.core.reference.coordinate_map`. Its constructor takes two
   coordinate has a signature *(mapping, input_coords(=domain),
   output_coords(=range))* along with an optional argument *inverse_mapping*
   specifying the inverse of *mapping*. This is a slightly different order from
   the :math:`(D, R, f)` order of this document. As noted above, the tuple
   :math:`(D, R, f)` has some redundancy because the function :math:`f` must
   know its domain, and, implicitly its range.  In :mod:`numpy`, it is
   impractical to really pass :math:`f` to the constructor because :math:`f`
   would expect something of *dtype* :math:`D` and should return someting of
   *dtype* :math:`R`. Therefore, *mapping* is actually a callable that
   represents the function :math:`\tilde{f} = I_R \circ f \circ I_D^{-1}`. Of
   course, the function :math:`f` can be recovered as :math:`f` = I_R^{-1} \circ
   \tilde{f} I_D`. In code, :math:`f` is roughly equivalent to:

   >>> domain = coordmap.function_domain
   >>> range = coordmap.function_range
   >>> f_tilde = coordmap.mapping
   >>> in_dtype = domain.coord_dtype
   >>> out_dtype = range.dtype

   >>> def f(d):
   ...    return f_tilde(d.view(in_dtype)).view(out_dtype)


The class *CoordinateMap* has an *inverse* property and there are module level
functions called *product, compose, linearize* and it has methods
*reordered_input, reordered_output*.

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

newdata = data.transpose([2,0,1])

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
