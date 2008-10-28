import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import visualization, io # FIAC.visualization
import spm_fixed

## Just need to change the output path

class Fixed(visualization.Fixed):

    def resultpath(self, path):
        return os.path.join(self.root, 'fixed-spm', self.design, self.which, self.contrast, path)

## No changes to run, except Fixed class

def run(contrast='average', which='contrasts', design='event'):
    for stat in ['effect', 'sd', 't']:
        v = Fixed(root=io.data_path,
                  stat=stat,
                  which=which,
                  contrast=contrast,
                  design=design)
        if stat == 't':
            v.vmax = 4.5; v.vmin = -4.5
        v.draw()
        v.output()
        
        htmlfile = file(v.resultpath("index.html"), 'w')
        htmlfile.write("""
        <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
        <html> <head>
        <title></title>
        </head>
        
        <body>
        <h2>Contrast %s, %s design, %s</h2>
        <h3>Effect</h3>
        <img src="effect.png">
        <h3>SD</h3>
        <img src="sd.png">
        <h3>T</h3>
        <img src="t.png">
        </body>
        </html>
        """ % (contrast, design, {'contrasts': 'magnitude', 'delays':'delay'}[which]))
        htmlfile.close()
    del(v)
