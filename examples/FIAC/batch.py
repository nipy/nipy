import os, time, string

ttoc = time.time()
for subj in range(16):

    os.system('python2.4 ./subject.py %d' % subj)
    ttic = time.time()

    print 'total time for %d subjects (minutes): %02f' % ((subj+1), ((ttic-ttoc)/60))

ttoc = time.time()

for which in ['delays', 'contrasts']:
    for design in ['block', 'event']:
        for contrast in ['overall', 'speaker', 'sentence', 'interaction']:
            for stat in ['t', 'sd', 'effect']:

                os.system('python2.4 ./plots-fixed.py --which=%s --contrast=%s --design=%s --stat=%s' %(which, contrast, design, stat))
                print 'plots done', which, contrast, design, stat
ttic = time.time()
print 'total time for fixed plots : %02f' % ((ttic-ttoc)/60)

ttoc = time.time()
for what in ['rho', 'fwhmOLS']:
    os.system('python2.4 ./plots-run.py --what=%s' % what)
ttic = time.time()
print 'total time for runs plots : %02f' % ((ttic-ttoc)/60)

ttoc = time.time()
for which in ['delays', 'contrasts']:
    for design in ['block', 'event']:
        for contrast in ['overall', 'speaker', 'sentence', 'interaction']:
            cmd = """
            python2.4 ./multi.py --which=%s
            --design=%s --contrast=%s --clobber
            """ % (which, design, contrast)
            cmd = string.join(cmd.replace('\n', ' ').strip().split())
            print cmd
            os.system(cmd)
ttic = time.time()
print 'total time for fixed effects group analysis : %02f' % ((ttic-ttoc)/60)


