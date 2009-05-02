import numpy as np
import fff2.nonparametric.twosample as f2

n1 = 8
n2 = 8

y1 = np.random.rand(n1)
v1 = .1*np.random.rand(n1)

y2 = np.random.rand(n2)
v2 = .1*np.random.rand(n2)

nperms = f2.count_permutations(n1, n2)

magics = np.asarray(range(nperms))

#t = f2.stat(y1,y2,id='student',Magics=magics)
t = f2.stat_mfx(y1,v1,y2,v2,id='student_mfx',Magics=magics)

from pylab import * 
hist(t,101)
show()
