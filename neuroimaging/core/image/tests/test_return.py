from neuroimaging.core.api import load_image
from neuroimaging.testing import anatfile, funcfile

anat = load_image(anatfile)
func = load_image(funcfile)

x = anat[0]
print x.grid.mapping.transform
print [a.name for a in x.grid.input_coords.axes()]
print [a.name for a in x.grid.output_coords.axes()]
print [a.name for a in anat.grid.output_coords.axes()]
print anat.grid.mapping.transform

print anat[2:5].shape
print anat.shape
print anat[2:10:2,0:2,0:5,4:8].shape


