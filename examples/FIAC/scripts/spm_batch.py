import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fiac, io

## for i in fiac.subjects[:-1]:
##     for run in range(1, 5):
##         cmd = "python2.5 spmmodel_run.py %d %d" % (i, run)
##         os.system(cmd)
##     cmd = "python2.5 spmfixed_run.py %d" % (i,)
##     os.system(cmd)

import spm_model.spm_visualization as visualization

for contrast in ['average', 'interaction', 'speaker', 'sentence']:
    for which in ['contrasts', 'delays']:
        for design in ['event', 'block']:

            cmd = "python2.5 spmvisualization_run.py %s %s %s" % (design, contrast, which)

            cmd += """
            rm %(p)s/fixed-spm/%(d)s/%(w)s/%(c)s/*/t.nii;
            rm %(p)s/fixed-spm/%(d)s/%(w)s/%(c)s/*/sd.nii;
            rm %(p)s/fixed-spm/%(d)s/%(w)s/%(c)s/*/effect.nii;
            """ % {'p': io.data_path, 'd':design, 'c':contrast, 'w':which}
            os.system(cmd)
            print "http://kff.stanford.edu/FIAC/fixed/%s/%s/%s" % (design, which, contrast)
