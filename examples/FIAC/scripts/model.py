if __name__ == '__main__':

    import sys
    if len(sys.argv) == 3:
        subj, run = map(int, sys.argv[1:])
    else:
        subj, run = (3, 3)

    study = StudyModel(root=io.data_path)
    subject = SubjectModel(subj, study=study)
    runmodel = RunModel(subject, run)
#    runmodel.OLS(clobber=True)
#    runmodel.AR(clobber=True)

##    run.view()

##     c = contrasts(run)
##     for i in range(len(c)-1): # don't plot delay
##         pylab.figure()
##         c[i].view()

##    pylab.show()
