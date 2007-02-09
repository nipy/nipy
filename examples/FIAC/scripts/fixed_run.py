"""
Perform a fixed run of FIAC model

python fixed_run.py [subject]

"""

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import fixed, io

    if len(sys.argv) == 2:
        subject = int(sys.argv[1])
    else:
        subject = 3

    fixed.run(subj=subject)
    os.system("bzip2 /home/analysis/FIAC/fixed/*/*/*/fiac%d/*nii" % subject)
