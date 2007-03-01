import csv, sys, os
import numpy as N

sys.path.insert(0, "..")

import fiac, fixed, compare, io

def fixedcor(subj=15):

    results = []
    for contrast in ['average', 'interaction', 'speaker', 'sentence']:
        for which in ['contrasts', 'delays']:
            for design in ['event', 'block']:

                study = fixed.Fixed(root=io.data_path, which=which, contrast=contrast, design=design)
                ni = fixed.Subject(subj, study)
                fmristat = compare.FixedSubject(subj, study)
                    
                _ni = ni.estimates(); _fmristat = fmristat.estimates()
                
                for i in range(3):
                    c = compare._cor(_ni[i], _fmristat[i], mask=N.ones(_ni[i].shape))
                    
                    print (subj, design, contrast, which, ['effect', 'sd', 't'][i], c)
                    results.append((subj, design, contrast, which, ['effect', 'sd', 't'][i], c))

                [os.system("bzip2 %s" % ni.resultpath("%s.nii" % stat)) for stat in ['t', 'effect', 'sd']]
                [os.system("rm %s" % ni.resultpath("%s.nii" % stat)) for stat in ['t', 'effect', 'sd']]


    return results

ofile = file("correlations_fixed.csv", "w")
writer = csv.writer(ofile)

for i in fiac.subjects:
    results = fixedcor(subj=i)
    for row in results:
        writer.writerow(row)
ofile.close()
                        
