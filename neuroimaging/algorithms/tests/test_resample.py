
import numpy as np
from neuroimaging.core.api import Affine, Image, SamplingGrid, Mapping

from neuroimaging.algorithms.resample import resample, affine

def test_resample2d():

    g = SamplingGrid.from_affine(Affine(np.diag([0.5,0.5,1])), ['x', 'y'], (100,90))
    i = Image(np.ones((100,90)), g)
    i[50:55,40:55] = 3.
    
    # This mapping describes a mapping from the "target" physical coordinates
    # to the "image" physical coordinates

    # The 3x3 matrix below indicates that the "target" physical coordinates
    # are related to the "image" physical coordinates by a shift of
    # -4 in each coordinate.

    # Or, to find the "image" physical coordinates, given the "target" physical
    # coordinates, we add 4 to each "target coordinate"

    # The resulting resampled image should show the overall image shifted
    # -8,-8 voxels towards the origin

    a = np.identity(3)
    a[:2,-1] = 4.

    ir = resample(i, i.grid, Affine(a))
    assert(np.allclose(ir[42:47,32:47], 3.))

    return i, ir

def test_resample2d1():

    g = SamplingGrid.from_affine(Affine(np.diag([0.5,0.5,1])), ['x', 'y'], (100,90))
    i = Image(np.ones((100,90)), g)
    i[50:55,40:55] = 3.
    
    a = np.identity(3)
    a[:2,-1] = 4.

    A = np.identity(2)
    b = np.ones(2)*4
    def mapper(x):
        return np.dot(A, x) + np.multiply.outer(b, np.ones(x.shape[1:]))
    ir = resample(i, i.grid, mapper)
    assert(np.allclose(ir[42:47,32:47], 3.))

    return i, ir

def test_resample2d2():

    g = SamplingGrid.from_affine(Affine(np.diag([0.5,0.5,1])), ['x', 'y'], (100,90))
    i = Image(np.ones((100,90)), g)
    i[50:55,40:55] = 3.
    
    a = np.identity(3)
    a[:2,-1] = 4.

    A = np.identity(2)
    b = np.ones(2)*4
    ir = affine(i, i.grid, (A, b))
    assert(np.allclose(ir[42:47,32:47], 3.))

    return i, ir
def test_resample2d3():

    g = SamplingGrid.from_affine(Affine(np.diag([0.5,0.5,1])), ['x', 'y'], (100,90))
    i = Image(np.ones((100,90)), g)
    i[50:55,40:55] = 3.
    
    a = np.identity(3)
    a[:2,-1] = 4.

    ir = affine(i, i.grid, a)
    assert(np.allclose(ir[42:47,32:47], 3.))

    return i, ir
    

def test_resample3d():

    g = SamplingGrid.from_affine(Affine(np.diag([0.5,0.5,0.5,1])), ['x', 'y', 'z'],
                                 (100,90,80))
    i = Image(np.ones(g.shape), g)
    i[50:55,40:55,30:33] = 3.
    
    # This mapping describes a mapping from the "target" physical coordinates
    # to the "image" physical coordinates

    # The 4x4 matrix below indicates that the "target" physical coordinates
    # are related to the "image" physical coordinates by a shift of
    # -4 in each coordinate.

    # Or, to find the "image" physical coordinates, given the "target" physical
    # coordinates, we add 4 to each "target coordinate"

    # The resulting resampled image should show the overall image shifted 
    # [-6,-8,-10] voxels towards the origin

    a = np.identity(4)
    a[:3,-1] = [3,4,5]

    ir = resample(i, i.grid, Affine(a))
    assert(np.allclose(ir[44:49,32:47,20:23], 3.))

def test_nonaffine():
    """
    This resamples an image along a curve through the image.

    """
    
    g = SamplingGrid.from_affine(Affine(np.identity(3)), ['x', 'y'], (100,90))
    i = Image(np.ones((100,90)), g)
    i[50:55,40:55] = 3.


    tgrid = SamplingGrid.from_start_step(['t'], [0], [np.pi*1.8/100], (100,))
    def curve(x):
        return (np.vstack([5*np.sin(x),5*np.cos(x)]).T + [52,47]).T

    m = Mapping(curve, tgrid.output_coords, i.grid.output_coords)
    ir = resample(i, tgrid, m)

    pylab.figure(num=3)
    pylab.imshow(i, interpolation='nearest')
    d = curve(np.linspace(0,1.8*np.pi,100))
    pylab.plot(d[0], d[1])
    pylab.gca().set_ylim([0,99])
    pylab.gca().set_xlim([0,89])

    pylab.figure(num=4)
    pylab.plot(np.asarray(ir))
    
def test_nonaffine2():
    """
    This resamples an image along a curve through the image.

    """
    
    g = SamplingGrid.from_affine(Affine(np.identity(3)), ['x', 'y'], (100,90))
    i = Image(np.ones((100,90)), g)
    i[50:55,40:55] = 3.


    tgrid = SamplingGrid.from_start_step(['t'], [0], [np.pi*1.8/100], (100,))
    print tgrid.range()
    print choke
    def curve(x):
        return (np.vstack([5*np.sin(x),5*np.cos(x)]).T + [52,47]).T

    ir = resample(i, tgrid, curve)

    pylab.figure(num=5)
    pylab.plot(np.asarray(ir))
    

if __name__ == "__main__":
    a, b = test_resample2d()
    test_resample2d1()
    test_resample2d2()
    test_resample2d3()

    import pylab

    pylab.figure(num=1)
    pylab.imshow(a, interpolation='nearest')
    pylab.colorbar()

    pylab.figure(num=2)
    pylab.imshow(b, interpolation='nearest')
    pylab.colorbar()

    test_resample3d()
    test_nonaffine()
    test_nonaffine2()
    pylab.show()
