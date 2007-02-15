if __name__ == '__main__':

    import sys
    if len(sys.argv) == 3:
        subj, run = map(int, sys.argv[1:])
    else:
        subj, run = (3, 3)

    study = model.StudyModel(root=io.data_path)
    subject = model.SubjectModel(subj, study=study)

##     runmodel = CompareRun(subject, run)
##     print runmodel.type

##     try:
##         runmodel.OLS(clobber=True)
##         runmodel.AR(clobber=True)
##         runmodel.max_corr()
##         runmodel.webpage()
##     except (urllib2.HTTPError, ValueError, NotImplementedError):
##         pass

##     print runmodel.corr
##     runmodel.load()
##     mask = runmodel.mask


##     vmodel = VoxelModel(subject, run)
##     vmodel.OLS(clobber=True)
##     vmodel.voxel_model(15,32,20)


      

    runmodel = CompareRun(subject, run)
    print runmodel.resultdir
##    keithrho = KeithRho(subject, run, resultdir=os.path.join("fsl", "fmristat_rho"))
    keithrho = runmodel
    print keithrho.type
    keithrho.OLS(clobber=True)
    keithrho.AR(clobber=True)

    M = keithrho
    print M.__class__
    M.load()
    corr = {}
    for stat in ['t', 'sd', 'effect']:
        corr[stat] = N.zeros((4,4))
        mask = M.mask[:].astype(N.bool)
        for i in range(4):
            for j in range(4):
                c1 = ['average', 'sentence', 'speaker', 'interaction'][i]
                c2 = ['average', 'sentence', 'speaker', 'interaction'][j]

                x1 = M.result(stat=stat, contrast=c1, which='delays')
                x2 = M.keith_result(stat=stat, contrast=c2, which='delays')

                x1 = N.nan_to_num(x1[:])[mask]
                x2 = N.nan_to_num(x2[:])[mask]

                x1.shape = N.product(x1.shape)
                x2.shape = N.product(x2.shape)

                corr[stat][i,j] = N.corrcoef(x1, x2)[0,1]
        print N.diagonal(corr[stat])

#    plot_result(M.result, M.keith_result, M.mask, contrast='speaker', which='delays', stat='t')


