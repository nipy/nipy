def FIACmultiview(**opts):
    i = neuroimaging.core.image.Image(FIACmultipath(**opts))
    i.grid = standard.grid
    v = viewer.BoxViewer(v)
    v.draw()
    pylab.show()

def FIACmultipath(**opts):
    return '/home/analysis/FIAC/multi/%(design)s/%(which)s/%(contrast)s/%(stat)s.img' % opts

