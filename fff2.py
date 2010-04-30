from nipy.neurospin import * 
import nipy.neurospin
__doc__ = nipy.neurospin.__doc__

# The following piece of code is ugly but is here to allow for
# statements like "import fff2.subpackage" or "from fff2.subpackage
# import something".

import sys, os 

ni = 'nipy'
ns = 'neurospin'
fff2 = __import__('fff2')
sys.modules['fff2'] = fff2

pydir = os.path.split(fff2.__file__)[0]
subpackages = os.listdir(os.path.join(pydir, ni, ns))

for sub in subpackages: 
    if os.path.isdir(os.path.join(pydir, ni, ns, sub)):
        submodule = __import__('nipy.neurospin.'+sub, 
                               fromlist=['neurospin.'+sub])
        sys.modules['fff2.'+sub] = submodule
        fff2.__setattr__(sub, submodule)
        
