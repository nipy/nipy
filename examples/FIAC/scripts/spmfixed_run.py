"""
Perform a fixed run of FIAC model

python fixed_run.py [subject]

"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
from spm_model.spm_fixed import run as _run

def run(subject):
    os.system("rm %s/fixed-spm/*/*/*/fiac%d/*bz2" % (io.data_path, subject))

    try:
        _run(subj=subject)
    except:
        pass
    os.system("bzip2 %s/fixed-spm/*/*/*/fiac%d/*nii" % (io.data_path, subject))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        subject = int(sys.argv[1])
    else:
        subject = 1


    run(subj=subject)


