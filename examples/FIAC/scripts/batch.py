import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fiac, io

## for i in fiac.subjects[-1:]:
##     for run in range(1, 5):
##         cmd = "python model_run.py %d %d" % (i, run)
##         os.system(cmd)
##     cmd = "python fixed_run.py %d" % (i,)
##     os.system(cmd)

import visualization, compare
for contrast in ['average', 'interaction', 'speaker', 'sentence']:
    for which in ['contrasts', 'delays']:
        for design in ['event', 'block']:

            visualization.run(contrast=contrast,
                              which=which,
                              design=design)

            compare.visualization_run(contrast=contrast,
                                       which=which,
                                       design=design)

            cmd = """
            rm %(p)s/fixed/%(d)s/%(w)s/%(c)s/*/t.nii;
            rm %(p)s/fixed/%(d)s/%(w)s/%(c)s/*/sd.nii;
            rm %(p)s/fixed/%(d)s/%(w)s/%(c)s/*/effect.nii;
            """ % {'p': io.data_path, 'd':design, 'c':contrast, 'w':which}
            os.system(cmd)
            print "http://kff.stanford.edu/FIAC/fixed/%s/%s/%s" % (design, which, contrast)
