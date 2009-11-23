""" test code to make a design matrix
"""
import numpy as np
from nipy.neurospin.utils.design_matrix import dmtx_light


tr = 1.0
frametimes = np.linspace(0,127*tr,128)
conditions = [0,0,0,1,1,1,3,3,3]
onsets=[30,70,100,10,30,90,30,40,60]
hrf_model = 'Canonical'
motion = np.cumsum(np.random.randn(128,6),0)
add_reg_names = ['tx','ty','tz','rx','ry','rz']

#event-related design matrix
paradigm = np.vstack(([conditions, onsets])).T
x1,name1 = dmtx_light(frametimes, paradigm, drift_model='Polynomial',
                      drift_order=3, add_regs=motion, add_reg_names=add_reg_names)

# block design matrix
duration = 7*np.ones(9)
paradigm = np.vstack(([conditions, onsets, duration])).T
x2,name2 = dmtx_light(frametimes, paradigm, drift_model='Polynomial', drift_order=3)

# FIR model
paradigm = np.vstack(([conditions, onsets])).T
hrf_model = 'FIR'
x3,name3 = dmtx_light(frametimes, paradigm, hrf_model = 'FIR',
                      drift_model='Polynomial', drift_order=3,
                      fir_delays = range(1,6))

import matplotlib.pylab as mp
mp.figure()
mp.imshow(x1/np.sqrt(np.sum(x1**2,0)),interpolation='Nearest', aspect='auto')
mp.xlabel('conditions')
mp.ylabel('scan number')
if name1!=None:
    mp.xticks(np.arange(len(name1)),name1,rotation=60,ha='right')
    mp.subplots_adjust(top=0.95,bottom=0.25)
mp.title('Example of event-related design matrix')

mp.figure()
mp.imshow(x2/np.sqrt(np.sum(x2**2,0)),interpolation='Nearest', aspect='auto')
mp.xlabel('conditions')
mp.ylabel('scan number')
if name2!=None:
    mp.xticks(np.arange(len(name2)),name2,rotation=60,ha='right')
    mp.subplots_adjust(top=0.95,bottom=0.25)
mp.title('Example of block design matrix')

mp.figure()
mp.imshow(x3/np.sqrt(np.sum(x3**2,0)),interpolation='Nearest', aspect='auto')
mp.xlabel('conditions')
mp.ylabel('scan number')
if name3!=None:
    mp.xticks(np.arange(len(name3)),name3,rotation=60,ha='right')
    mp.subplots_adjust(top=0.95,bottom=0.25)
mp.title('Example of FIR design matrix')


mp.show()

