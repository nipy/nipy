from .viz import plot_map, cm

# Import here only files that don't draw in compiled code: that way the
# basic functionality is still usable even if the compiled
# code is messed up (32/64 bit issues, or binary incompatibilities)
from .datasets import as_volume_img, save
