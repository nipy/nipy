import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fiac

for i in fiac.subjects:
    for run in range(1,5):
        cmd = "python model_run.py %d %d" % (i, run)
        os.system(cmd)
    cmd = "python fixed_run.py %d" % (i,)
    os.system(cmd)

