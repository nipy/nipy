import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fiac

for i in fiac.subjects[-1:]:
    for run in range(1,5):
        cmd = "python model_run.py %d %d" % (i, run)
        os.system(cmd)
    cmd = "python fixed_run.py %d" % (i,)
    os.system(cmd)

import visualization
for contrast in ['average', 'interaction' ,'speaker', 'sentence']:
    for which in ['contrasts', 'delays']:
        for design in ['event', 'block']:
            print "http://kff.stanford.edu/FIAC/fixed/%s/%s/%s" % (design, which, contrast)
            visualization.run(contrast=contrast,
                              which=which,
                              design=design)

            cmd = """
            rm /home/analysis/FIAC/fixed/%(d)s/%(w)s/%(c)s/*/t.nii;
            rm /home/analysis/FIAC/fixed/%(d)s/%(w)s/%(c)s/*/sd.nii;
            rm /home/analysis/FIAC/fixed/%(d)s/%(w)s/%(c)s/*/effect.nii;
            """ % {'d':design, 'c':contrast, 'w':which}
            os.system(cmd)
