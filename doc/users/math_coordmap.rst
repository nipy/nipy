.. _math-coordmap:

**********************************************
Mathematical formulation of the Coordinate Map
**********************************************

Using the *CoordinateMap* can be a little hard to get used to.  For some users,
a mathematical description, free of any python syntax and code design and
snippets may be helpful. After following through this description, the code
design and usage may be clearer.

We return to the normalization example in :ref:`normalize-coordmap` and try to
write it out mathematically.  Conceptually, to do normalization, we need to be
able to answer each of these three questions:

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

For more detail on the ideas behind the coordmap design, see
:ref:``coordmp-discussion`.
