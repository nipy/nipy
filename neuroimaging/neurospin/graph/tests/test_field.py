#!/usr/bin/env python

from numpy.testing import *
import numpy as np
import neuroimaging.neurospin.graph.field as ff

def basic_field():
    dx = 10
    dy = 10
    dz = 10
    F = ff.Field(dx*dy*dz)
    
    XYZ = np.array( [[x,y,z] for z in range(dz) for y in range(dy) for x in range(dx)] )
    F.from_3d_grid(XYZ,26)
    data = np.sum(XYZ,1).astype('d')
    F.set_field(data)
    return F

def basic_field_2():
    dx = 10
    dy = 10
    dz = 10
    F = ff.Field(dx*dy*dz)
    XYZ = np.array( [[x,y,z] for z in range(dz) for y in range(dy) for x in range(dx)] )
    toto = np.array([[x-5,y-5,z-5] for z in range(dz) for y in range(dy) for x in range(dx)] )

    data = np.sum(toto*toto,1) 
    F.from_3d_grid(XYZ,26)
    F.set_field(data)
    return F

def basic_graph():
    dx = 10
    dy = 10
    dz = 10
    XYZ = np.array( [[x,y,z] for z in range(dz) for y in range(dy) for x in range(dx)] )
    G = ff.Field(dx*dy*dz)
    G.from_3d_grid(XYZ,26);
    return G 



class test_Field(TestCase):

    def test_max_1(self):
        F  = basic_field()
        F.field[555] = 30
        depth = F.local_maxima()
        Dep = np.zeros(1000,'i')
        Dep[555] = 4; 
        Dep[999] = 3; 
        OK = sum(np.absolute(Dep-depth))<1.e-7
        self.assert_(OK)

    def test_max_2(self):
        F  = basic_field()
        F.field[555] = 28
        idx,depth = F.get_local_maxima()
        self.assert_(len(idx) == 2)
        self.assert_(np.alltrue( idx == (555, 999) ))
        self.assert_(np.alltrue( depth == (4, 3) ))

    def test_max_3(self):
        F  = basic_field()
        F.field[555] = 27
        idx,depth = F.get_local_maxima()
        OK = (np.size(idx)==2)&(idx[0]==555)&(idx[1]==999)&(depth[0]==6)&(depth[1]==6)
        self.assert_(OK)

    def test_max_4(self):
        F  = basic_field()
        F.field[555] = 28
        idx,depth = F.get_local_maxima(0,27)
        OK = (np.size(idx)==1)&(idx[0]==555)&(depth[0]==1)
        self.assert_(OK)
        

    def test_smooth_1(self):
        G  = basic_graph()
        field = np.zeros((1000,1))
        field[555,0] = 1
        G.set_field(field)
        G.diffusion()
        sfield = G.get_field()
        OK1 = (sfield[555]==0)
        OK2 = (sfield[554]==1)
        OK3 = (np.absolute(sfield[566]-np.sqrt(2))<1.e-7)
        OK4 = (np.absolute(sfield[446]-np.sqrt(3))<1.e-7)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_smooth_2(self):
        G  = basic_graph()
        field = np.zeros((1000,1))
        field[555,0] = 1
        G.set_field(field)
        G.diffusion(1)
        sfield = G.get_field()
        OK1 = (sfield[555]==0)
        OK2 = (sfield[554]==1)
        OK3 = (np.absolute(sfield[566]-np.sqrt(2))<1.e-7)
        OK4 = (np.absolute(sfield[446]-np.sqrt(3))<1.e-7)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_dilation(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.dilation(2)
        OK1 = (F.field[737]==30)
        OK2 = (F.field[0]==6)
        OK3 = (F.field[999]==27)
        OK4 = (F.field[664]==30)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_erosion(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.erosion(2)
        field = F.get_field()
        OK1 = (field[737]==11)
        OK2 = (field[0]==0)
        OK3 = (field[999]==21)
        OK4 = (field[664]==0)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_opening(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.opening(2)
        field = F.get_field()
        OK1 = (field[737]==17)
        OK2 = (field[0]==0)
        OK3 = (field[999]==21)
        OK4 = (field[555]==16)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_closing(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.closing(2)
        field = F.get_field()
        OK1 = (field[737]==17)
        OK2 = (field[0]==6)
        OK3 = (field[999]==27)
        OK4 = (field[555]==30)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_watershed_1(self):
        F = basic_field()
        F.field[555] = 28
        F.field[664] = 0
        #F.field = np.reshape(F.field,(1,f.V))
        idx,depth, major,label = F.custom_watershed()
        OK1 = np.size(idx)==2
        OK2 = (idx[0]==555)&(idx[1]==999)
        OK3 = (major[0]==0)&(major[1]==0)
        OK4 = (label[123]==0)&(label[776]==1)&(label[666]==0)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_watershed_2(self):
        F = basic_field_2()
        F.field[555] = 10
        F.field[664] = 0
        idx,depth, major,label = F.custom_watershed()
        OK1 = np.size(idx)==9
        OK3 = (major[0]==0)&(major[6]==0)
        OK = OK1 & OK3
        self.assert_(OK)

    def test_watershed_3(self):
        F  = basic_field_2()
        F.field[555] = 10
        F.field[664] = 0
        idx,depth, major,label = F.custom_watershed(0,11)
        OK = np.size(idx)==8
        self.assert_(OK)

    def test_bifurcations_1(self):
        F = basic_field()   
        idx,height, parent,label = F.threshold_bifurcations()
        OK1= (idx==999)
        OK2= (parent==0);
        OK = OK1 & OK2
        self.assert_(OK)

    def test_bifurcations_2(self):
        F = basic_field_2()
        idx,height, parent,label = F.threshold_bifurcations()
        OK = np.size(idx==15)
        self.assert_(OK)
    
if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])

