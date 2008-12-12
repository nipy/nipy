"""
Coordinate maps store all the details about how an image translates to space.
They also provide mechanisms for iterating over that space.
"""
import copy, string

import numpy as np

from neuroimaging.core.reference.axis import Axis
from neuroimaging.core.reference.coordinate_system import CoordinateSystem
from neuroimaging.core.reference.mapping import Mapping, Affine

__docformat__ = 'restructuredtext'

class CoordinateMap(object):
    """
    Defines a set of input and output coordinate systems and a mapping
    between the two, which represents the mapping of (for example) an
    image from voxel space to real space.
    """
    
    @staticmethod
    def from_start_step(innames, outnames, start, step, shape):
        """
        Create a `CoordinateMap` instance from sequences of names, shape, start
        and step.

        :Parameters:
            innames : ``tuple`` of ``string``
                TODO
            outnames : ``tuple`` of ``string``
                TODO
            start : ``tuple`` of ``float``
                TODO
            step : ``tuple`` of ``float``
                TODO
            shape: ''tuple'' of ''int''

        :Returns: `CoordinateMap`
        
        :Predcondition: ``len(names) == len(shape) == len(start) == len(step)``
        """
        ndim = len(innames)
        if len(outnames) != ndim:
            raise ValueError, 'len(innames) != len(outnames)'

        cmaps = []
        for i in range(ndim):
            A = np.array([[step[i], start[i]],
                          [0, 1]])
            cmaps.append(CoordinateMap.from_affine([innames[i]], [outnames[i]], Affine(A), (shape[i],)))
        return product(*cmaps)

    @staticmethod
    def identity(names, shape):
        """
        Return an identity coordmap of the given shape.
        
        :Parameters:
            shape : ``tuple`` of ``int``
                TODO
            names : ``tuple`` of ``string``
                TODO

        :Returns: `CoordinateMap` with `CoordinateSystem` input
                  and an identity transform 
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        return CoordinateMap.from_start_step(names, names, [0]*len(names),
                                             [1]*len(names), shape)

    @staticmethod
    def from_affine(innames, outnames, mapping, shape):
        """
        Return coordmap using a given `Affine` mapping
        
        :Parameters:
            innames : ``tuple`` of ``string``
                The names of the axes of the input coordinate systems

            outnames : ``tuple`` of ``string``
                The names of the axes of the output coordinate systems

            mapping : `Affine`
                An affine mapping between the input and output coordinate systems.
            shape : ''tuple'' of ''int''
                The shape of the coordmap
        :Returns: `CoordinateMap`
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        ndim = (len(innames) + 1, len(outnames) + 1)
        if mapping.transform.shape != ndim:
            raise ValueError('shape and number of axis names do not agree')
        A = Affine(mapping.transform) # NOTE: this Affine's matrix
                                      # will be either a 'float' or 'complex'
                                      # dtype
        dtype = A.transform.dtype
        inaxes = [Axis(name, length=l, dtype=dtype) for name, l in zip(innames, shape)]
        outaxes = [Axis(name, dtype=dtype) for name in outnames]
        input_coords = CoordinateSystem("input", inaxes)
        output_coords = CoordinateSystem('output', outaxes)
        return CoordinateMap(A, input_coords, output_coords)

    def _getinverse(self):
        """
        Return the inverse coordinate map.
        """
        if self.mapping.isinvertible:
            return CoordinateMap(self.mapping.inverse(), self.output_coords, self.input_coords)
    inverse = property(_getinverse)

    def __init__(self, mapping, input_coords, output_coords):
        """
        :Parameters:
            mapping : `mapping.Mapping`
                The mapping between input and output coordinates
            input_coords : `CoordinateSystem`
                The input coordinate system
            output_coords : `CoordinateSystem`
                The output coordinate system
        """
        # These guys define the structure of the coordmap.
        self.mapping = mapping
        self.input_coords = input_coords
        self.output_coords = output_coords

    def _getshape(self):
        return tuple([len(a) for a in self.input_coords.axes])
    shape = property(_getshape)

    def _getndim(self):
        return (len(self.input_coords.axes), len(self.output_coords.axes))
    ndim = property(_getndim)

    def isaffine(self):
        if isinstance(self.mapping, Affine):
            return True
        return False

    def _getaffine(self):
        if hasattr(self.mapping, "transform"):
            return self.mapping.transform
        raise AttributeError
    affine = property(_getaffine)

    def __call__(self, x, typecast=True):
        """
        Return self.mapping(x)

        If typecast, then return a view of the result with dtype of self.output_coords.dtype.

        >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
        >>> inc = CoordinateSystem('input', inaxes)
        >>> outaxes = [Axis(x) for x in 'xyz']
        >>> outc = CoordinateSystem('output', outaxes)
        >>> cm = CoordinateMap(Affine(np.diag([1,2,3,1])), inc, outc)
        >>> cm([2,3,4], view=True)
        array([(2.0, 6.0, 12.0)], dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        >>> cmi = cm.inverse
        >>> cmi([2,6,12], view=True)
        array([(2.0, 3.0, 4.0)], dtype=[('i', '<f8'), ('j', '<f8'), ('k', '<f8')])
        >>>                                    
        """
        x = self.input_coords.typecast(x, dtype=self.input_coords.dtype)
        y = self.mapping(x)
        if typecast:
            y = self.output_coords.typecast(y, dtype=self.output_coords.dtype)
        if len(x.shape) in [0,1]:
            y = np.squeeze(y)
        return y
#         if view:
#             # Need to copy the transposed data
#             # for the order of the data to be correct
#             # for np.recarray
#             yc = np.array(y.T, copy=True, order='C')
#             y = np.recarray(buf=yc, dtype=self.output_coords.dtype, shape=np.product(y.shape[1:]))

#         return y

    def copy(self):
        """
        Create a copy of the coordmap.

        :Returns: `CoordinateMap`
        """
        return CoordinateMap(self.mapping, self.input_coords,
                            self.output_coords)

    def __getitem__(self, index):
        """
        If all input coordinates have a len, return
        a slice through the coordmap.

        Parameters
        ----------
        index : ``int`` or ``slice``
            sequence of integers or slices
        
        """

        varcoords, mapping, shape = self.mapping._slice_mapping(index, self.shape)
        ia = self.input_coords.axes
        newia = []
        for i in range(self.ndim[0]):
            if i in varcoords:
                a = copy.deepcopy(ia[i])
                newia.append(a)
        newic = CoordinateSystem(self.input_coords.name, newia, shape=shape)
        return CoordinateMap(mapping, newic, self.output_coords)

    def range(self):
        """
        Return the coordinate values in the same format as numpy.indices.
        
        :Returns: TODO
        """
      
        indices = np.indices(self.shape)
        tmp_shape = indices.shape
            # reshape indices to be a sequence of coordinates
        indices.shape = (self.ndim[0], np.product(self.shape))
        _range = self.mapping(indices)
        _range.shape = tmp_shape
        return _range 


def rename_input(coordmap, **kwargs):
    """
    Rename the input_coords, returning a new CoordinateMap

    >>> import numpy as np
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> print cm.input_coords.values()
    [<Axis:"i", dtype=[('i', '<f8')], length=10>, <Axis:"j", dtype=[('j', '<f8')], length=20>, <Axis:"k", dtype=[('k', '<f8')], length=30>]
    >>> cm2 = rename_input(cm, i='x')
    >>> print cm2.input_coords
    {'axes': [<Axis:"x", dtype=[('x', '<f8')], length=10>, <Axis:"j", dtype=[('j', '<f8')], length=20>, <Axis:"k", dtype=[('k', '<f8')], length=30>], 'name': 'input-renamed'}
        
    """
    input_coords = coordmap.input_coords.rename(**kwargs)
    return CoordinateMap(coordmap.mapping, input_coords, coordmap.output_coords)

def rename_output(coordmap, **kwargs):
    """
    Rename the output_coords, returning a new CoordinateMap.
    
    >>> import numpy as np
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> print cm.output_coords.values()
    [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"y", dtype=[('y', '<f8')]>, <Axis:"z", dtype=[('z', '<f8')]>]
    >>> cm2 = cm.rename_output(y='a')
    >>> print cm2.output_coords
    {'axes': [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"a", dtype=[('a', '<f8')]>, <Axis:"z", dtype=[('z', '<f8')]>], 'name': 'output-renamed'}

    >>>                             
    """
    output_coords = coordmap.output_coords.rename(**kwargs)
    return CoordinateMap(coordmap.mapping, coordmap.input_coords, output_coords)
        
def reorder_input(coordmap, order=None):
    """
    Create a new coordmap with reversed input_coords.
    Default behaviour is to reverse the order of the input_coords.
    If the coordmap has a shape, the resulting one will as well.

    Inputs:
    -------
    order: sequence
         Order to use, defaults to reverse. The elements
         can be integers, strings or 2-tuples of strings.
         If they are strings, they should be in coordmap.input_coords.axisnames.

    Returns:
    --------

    newcoordmap: `CoordinateMap`
         A new CoordinateMap with reversed input_coords.

    >>> inc = CoordinateSystem('input', inaxes)
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> reorder_input(cm, 'ikj').shape
    (10, 30, 20)

    """
    ndim = coordmap.ndim[0]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [coordmap.input_coords.axisnames.index(s) for s in order]

    newaxes = [coordmap.input_coords.axes[i] for i in order]
    newincoords = CoordinateSystem(coordmap.input_coords.name + '-reordered', newaxes)
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    A = CoordinateMap(Affine(perm), newincoords, coordmap.input_coords)
    return compose(coordmap, A)

def reorder_output(coordmap, order=None):
    """
    Create a new coordmap with reversed output_coords.
    Default behaviour is to reverse the order of the input_coords.
    
    Inputs:
    -------

    order: sequence
         Order to use, defaults to reverse. The elements
         can be integers, strings or 2-tuples of strings.
         If they are strings, they should be in coordmap.output_coords.axisnames.

    Returns:
    --------
        
    newcoordmap: `CoordinateMap`
         A new CoordinateMap with reversed output_coords.

    >>> inc = CoordinateSystem('input', inaxes)
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> reorder_output(cm, 'xzy').shape
    (10, 20, 30)
    >>> reorder_output(cm, [0,2,1]).shape
    (10, 20, 30)
    >>>                             

    >>> newcm = reorder_output(cm, 'yzx')
    >>> newcm.output_coords.axisnames
    ['y', 'z', 'x']
    >>>                              

    """

    ndim = coordmap.ndim[1]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [coordmap.output_coords.axisnames.index(s) for s in order]

    newaxes = [coordmap.output_coords.axes[i] for i in order]
    newoutcoords = CoordinateSystem(coordmap.output_coords.name + '-reordered', newaxes)
    
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    A = CoordinateMap(Affine(perm), coordmap.output_coords, newoutcoords)
    return compose(A, coordmap)

def product(*cmaps):
    """
    Return the "topological" product of two or more CoordinateMaps.

    Inputs:
    -------
    cmaps : sequence of CoordinateMaps

    Returns:
    --------
    cmap : ``CoordinateMap``

    >>> inc1 = CoordinateMap.from_affine('i', 'x', Affine(np.diag([2,1])), (10,))
    >>> inc2 = CoordinateMap.from_affine('j', 'y', Affine(np.diag([3,1])), (20,))
    >>> inc3 = CoordinateMap.from_affine('k', 'z', Affine(np.diag([4,1])), (30,))

    >>> cmap = product(inc1, inc3, inc2)
    >>> cmap.shape
    (10, 30, 20)
    >>> cmap.input_coords.axisnames
    ['i', 'k', 'j']
    >>> cmap.output_coords.axisnames
    ['x', 'z', 'y']
    >>> cmap.affine
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  4.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  1.]])

    """
    inaxes = []
    outaxes = []
    innames = []
    outnames = []
    mappings = []
    ndimin = []

    for cmap in cmaps:
        inaxes += cmap.input_coords.axes
        outaxes += cmap.output_coords.axes
        innames += [cmap.input_coords.name]
        outnames += [cmap.output_coords.name]
        mappings.append(cmap.mapping)
        ndimin.append(cmap.ndim[0])

    ndimin.insert(0,0)
    ndimin = tuple(np.cumsum(ndimin))
    innames = string.join(innames, ' * ')
    outnames = string.join(outnames, ' * ')

    def mapping(x):
        x = np.asarray(x)
        y = []
        for i in range(len(ndimin)-1):
            cmap = cmaps[i]
            yy = mappings[i](x[:,ndimin[i]:ndimin[i+1]])
            y.append(yy)
        yy = np.hstack(y)
        return yy

    notaffine = filter(lambda x: not isinstance(x, Affine), mappings)
    if not notaffine:
        Y = np.vstack([np.identity(ndimin[-1]), np.zeros(ndimin[-1])]).T
        d = mapping(Y.T).T
        dd = mapping(np.zeros((ndimin[-1]+1,ndimin[-1]))).T
        dd[:,-1] = 0.
        C = np.identity(d.shape[1]).astype(np.complex)
        C[:d.shape[0],:d.shape[1]] = d - dd
        if np.allclose(C.real, C):
            C = C.astype(np.float)
        mapping = Affine(C)
        for a in inaxes:
            a.dtype = mapping.transform.dtype
        for a in outaxes:
            a.dtype = mapping.transform.dtype

    return CoordinateMap(mapping, CoordinateSystem(innames, inaxes),
                         CoordinateSystem(outnames, outaxes))

def compose(*cmaps):
    """
    Return the (right) composition of two or more CoordinateMaps.

    Inputs:
    -------
    cmaps : sequence of CoordinateMaps

    Returns:
    --------
    cmap : ``CoordinateMap``
         The resulting CoordinateMap has input_coords == cmaps[-1].input_coords
         and output_coords == cmaps[0].output_coords

    >>> cmap = CoordinateMap.from_affine('i', 'x', Affine(np.diag([2,1])), (10,))
    >>> cmapi = cmap.inverse
    >>> id1 = compose(cmap,cmapi)
    >>> print id1.affine
    [[ 1.  0.]
     [ 0.  1.]]
    >>> assert not hasattr(id1, 'shape')
    >>> id2 = compose(cmapi,cmap)
    >>> assert id2.shape == (10,)
    >>> id1.input_coords.axisnames
    ['x']
    >>> id2.input_coords.axisnames
    ['i']
    >>> 

    """

    cmap = cmaps[-1]
    for m in cmaps[:-1]:
        if m.input_coords == cmap.output_coords:
            cmap = CoordinateMap(m.mapping * cmap.mapping, cmap.input_coords, m.output_coords)
        else:
            raise ValueError, 'input and output coordinates do not match: input=%s, output=%s' % (`m.input_coords.dtype`, `cmap.output_coords.dtype`)
    return cmap

def replicate(coordmap, n, concataxis='string'):
    """
    Create a CoordinateMap by taking the product
    of coordmap with a 1-dimensional 'concat' CoordinateSystem

    :Parameters:
         coordmap : `CoordinateMap`
                The coordmap to be used
         n : ``int``
                The number of tiems to concatenate the coordmap
         concataxis : ``string``
                The name of the new dimension formed by concatenation
    """
    concat = CoordinateMap.from_affine([concataxis], [concataxis], Affine(np.identity(2)), (n,))
    return product(concat, coordmap)

def hstack(*cmaps):
    """
    Return a "hstacked" CoordinateMap. That is,
    take the result of a number of cmaps, and return np.hstack(results)
    with an additional first row being the 'concat' axis values.

    If the cmaps are identical
    the resulting map is essentially
    replicate(cmaps[0], len(cmaps)) but the mapping is not Affine.

    Some simple modifications of this function would allow 'interpolation'
    along the 'concataxis'. 

    Inputs:
    -------
    cmaps : sequence of CoordinateMaps
          Each cmap should have the same input_coords, output_coords and shape.

    Returns:
    --------
    cmap : ``CoordinateMap``

    >>> inc1 = CoordinateMap.from_affine('ab', 'cd', Affine(np.diag([2,3,1])), (10,20))
    >>> inc2 = CoordinateMap.from_affine('ab', 'cd', Affine(np.diag([3,2,1])), (10,20))
    >>> inc3 = CoordinateMap.from_affine('ab', 'cd', Affine(np.diag([1,1,1])), (10,20))
    >>> stacked = hstack(inc1, inc2, inc3)

    >>> stacked(np.array([[0,1,2],[1,1,2],[2,1,2], [1,1,2]]).T)
    array([[ 0.,  2.,  6.],
           [ 1.,  3.,  4.],
           [ 2. , 1.,  2.],
           [ 1.,  3.,  4.]])
    >>> 

    """

    # Ensure that they all have the same coordinate systems

    notinput = filter(lambda i: cmaps[i].input_coords != cmaps[0].input_coords, range(len(cmaps)))
    notoutput = filter(lambda i: cmaps[i].output_coords != cmaps[0].output_coords, range(len(cmaps)))
    notshape = filter(lambda i: cmaps[i].shape != cmaps[0].shape, range(len(cmaps)))

    if notinput or notoutput or notshape:
        raise ValueError("input and output coordinates as well as shape of each CoordinateMap should be the same in order to stack them")

    def mapping(x, return_index=False):
        r = []
        for i in range(x.shape[1]):
            ii = int(x[0,i])
            y = cmaps[ii](x[1:,i])
            r.append(np.hstack([x[0,i], y]))
        return np.vstack(r)

    stackin = Axis('stack-input', length=len(cmaps))
    stackout = Axis('stack-output')

    inaxes = [stackin] + cmaps[0].input_coords.axes
    incoords = CoordinateSystem('stackin-%s' % cmaps[0].input_coords.name, 
                                inaxes)
    outaxes = [stackout] + cmaps[0].output_coords.axes
    outcoords = CoordinateSystem('stackout-%s' % cmaps[0].output_coords.name, 
                                 outaxes)
    return CoordinateMap(mapping, incoords, outcoords)

