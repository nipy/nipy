def plot_result(result1, result2, mask, which='contrasts', contrast='average', stat='t', vmax=None, vmin=None):
    
    resultargs = {'which':which, 'contrast':contrast, 'stat':stat}
    x = result1(**resultargs)
    y = result2(**resultargs)

    vx = BoxViewer(x, mask=mask, colormap='spectral')
    vy = BoxViewer(y, mask=mask, colormap='spectral')

    if vmin is not None:
        vx.m = vmin
        vy.m = vmin

    if vmax is not None:
        vx.M = vmax
        vy.M = vmax

    vx.draw()
    vy.draw()

    X = x.readall() * mask.readall()
    X.shape = N.product(X.shape)

    Y = y.readall() * mask.readall()
    Y.shape = N.product(Y.shape)

    print 'corr', N.corrcoef(X, Y)[0,1]
    pylab.show()



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


