"""
Perform a multi run of FIAC model

python multi_run.py [design, which, contrast]

"""

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import multi, io

    if len(sys.argv) == 4:
        design, which, contrast = sys.argv[1:]
    else:
        design, which, contrast = ('block', 'contrasts', 'average')

    m=multi.Multi(which=which, contrast=contrast, design=design, root=io.data_path)
    m.fit()
    os.system("bzip2 /home/analysis/FIAC/multi/%s/%s/%s/*nii" % (design, which, contrast))
