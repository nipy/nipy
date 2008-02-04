
import ipython1.kernel.api as kernel
rc = kernel.RemoteController(('127.0.0.1',10105))

rc.executeAll("""
import sys
sys.path.append("/home/jtaylo/jtaylo.stuff/code/nipy/examples/FIAC")""")


rc.executeAll("""
import scripts.model_run as model
import scripts.fixed_run as fixed
import scripts.spmmodel_run as spmmodel
import scripts.spmfixed_run as spmfixed
""")

## seems to be shared equally among 3 on my setup

run = rc.parallelize("all", "lambda x: [model.run(x, i+1) for i in range(4)]")
spmrun = rc.parallelize("all", "lambda x: [spmmodel.run(x, i+1) for i in range(4)]")

fixed = rc.parallelize("all", "lambda x: fixed.run(x)")
spmfixed = rc.parallelize("all", "lambda x: spmfixed.run(x)")



#[run(range(3*i,3*(i+1)) for i in range(6)]
[spmrun(range(3*i,3*(i+1))) for i in range(6)]
[fixed(range(3*i,3*(i+1))) for i in range(6)]
[spmfixed(range(3*i,3*(i+1))) for i in range(6)]

for contrast in ['average', 'interaction', 'speaker', 'sentence']:
    for which in ['contrasts', 'delays']:
        for design in ['event', 'block']:

            cmd = "python2.5 visualization_run.py %s %s %s;" % (design,
                                                               contrast,
                                                               which)
            cmd += """
            rm %(p)s/fixed/%(d)s/%(w)s/%(c)s/*/t.nii;
            rm %(p)s/fixed/%(d)s/%(w)s/%(c)s/*/sd.nii;
            rm %(p)s/fixed/%(d)s/%(w)s/%(c)s/*/effect.nii;
            """ % {'p': io.data_path, 'd':design, 'c':contrast, 'w':which}
            os.system(cmd)

            cmd = "python2.5 spmvisualization_run.py %s %s %s;" % (design,
                                                               contrast,
                                                               which)
            cmd += """
            rm %(p)s/fixed-spm/%(d)s/%(w)s/%(c)s/*/t.nii;
            rm %(p)s/fixed-spm/%(d)s/%(w)s/%(c)s/*/sd.nii;
            rm %(p)s/fixed-spm/%(d)s/%(w)s/%(c)s/*/effect.nii;
            """ % {'p': io.data_path, 'd':design, 'c':contrast, 'w':which}
            os.system(cmd)

            print "http://kff.stanford.edu/FIAC/fixed/%s/%s/%s" % (design, which, contrast)
