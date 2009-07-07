import numpy as np
import numpy.random as nr
import nipy.neurospin.graph.field as ff


dx = 50
dy = 50
dz = 1
nbseeds=10
F = ff.Field(dx*dy*dz)
xyz = np.reshape(np.indices((dx,dy,dz)),(3,dx*dy*dz)).T.astype(np.int)
F.from_3d_grid(xyz,18)
#data = 3*nr.randn(dx*dy*dz) + np.sum((xyz-xyz.mean(0))**2,1)
#F.set_field(np.reshape(data,(dx*dy*dz,1)))
data = nr.randn(dx*dy*dz,1)
F.set_weights(F.get_weights()/18)
F.set_field(data)
F.diffusion(5)
data = F.get_field()

seeds = np.argsort(nr.rand(F.V))[:nbseeds]
seeds, label, J0 = F.geodesic_kmeans(seeds)
wlabel, J1 = F.ward(nbseeds)
seeds, label, J2 = F.geodesic_kmeans(seeds,label=wlabel.copy(), eps = 1.e-7)

print 'inertia values for the 3 algorithms: ',J0,J1,J2

import matplotlib.pylab as mp
mp.figure()
mp.subplot(1,3,1)
mp.imshow(np.reshape(data,(dx,dy)),interpolation='nearest' )
mp.subplot(1,3,2)
mp.imshow(np.reshape(wlabel,(dx,dy)),interpolation='nearest' )
mp.subplot(1,3,3)
mp.imshow(np.reshape(label,(dx,dy)),interpolation='nearest' )
mp.show()
