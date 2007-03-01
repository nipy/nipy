"""
Compute and save correlation of effect, sd, t for each (subj, pair, contrast, mag/delay)
"""

import csv, sys, os

sys.path.insert(0, "..")

import fiac, model, compare, io

compare.Run.verbose = False
def runcor(subj=15, run=1):

    study = model.Study(root=io.data_path)
    subject = model.Subject(15, study=study)

    runmodel = compare.Run(subject, run)
    runmodel.load()

    results = []
    for contrast in ['average', 'interaction', 'speaker', 'sentence']:
        for which in ['contrasts', 'delays']:
            for stat in ['effect', 'sd', 't']:
                args = {'contrast':contrast,
                        'stat':stat, 
                        'which':which}
                c = compare._cor(runmodel.result(**args), runmodel.fmristat_result(**args), mask=runmodel.mask)
                print (subj, run, contrast, which, stat, c)
                os.system("bzip2 %s" % os.path.join(runmodel.resultdir, which, contrast,
                                                    "%s.nii" % stat))
                os.system("rm %s" % os.path.join(runmodel.resultdir, which, contrast,
                                                    "%s.nii" % stat))
                results.append((subj, run, contrast, which, stat, c))
    return results

ofile = file("correlations_runs.csv", "w")
writer = csv.writer(ofile)

for s in fiac.subjects[-1:] + fiac.subjects[0:-1]:
    results = []
    for i in range(1, 5):
        results += runcor(subj=s, run=i)
    for result in results:
        writer.writerow(result)
    ofile.flush()

ofile.close()
