"""
Compare design -- the LONG way -- making sure that final design
matrix in AR model agrees with fmristat
"""

import sys

sys.path.insert(0, "..")

import fiac, model, compare, io

compare.Run.verbose = False
def runcor(subj=15, run=1):

    study = model.Study(root=io.data_path)
    subject = model.Subject(subj, study=study)

    runmodel = compare.Run(subject, run)
    runmodel.load()

    runmodel.OLS(clobber=True)
    runmodel.AR(clobber=True)
    runmodel.max_corr()
    runmodel.webpage()



for s in fiac.subjects[-1:] + fiac.subjects[0:-1]:
    for i in range(1, 5):
        runcor(subj=s, run=i)

