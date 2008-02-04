"""
Perform a single run of FIAC model

python model_run.py [subject, run]

"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model, io
import spm_model.spm_model as spm

def run(subj, run):
    try:
        spm.run(subj=subj, run=run)
    except:
        pass
    os.system("bzip2 %s/fiac%d/fonc%d/spm/*/*/*nii" % (io.data_path, subj, run))

if __name__ == "__main__":
    if len(sys.argv) == 3:
        subj, run_ = map(int, sys.argv[1:])
    else:
        subj, run_ = (1, 2)

    run(subj, run_)

