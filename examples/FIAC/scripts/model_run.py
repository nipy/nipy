"""
Perform a single run of FIAC model

python model_run.py [subject, run]

"""

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import model, io

    if len(sys.argv) == 3:
        subj, run = map(int, sys.argv[1:])
    else:
        subj, run = (1, 2)

    model.run(subj=subj, run=run)
    os.system("bzip2 %s/fiac%d/fonc%d/fsl/fmristat_run/*/*/*nii" % (io.data_path, subj, run))
