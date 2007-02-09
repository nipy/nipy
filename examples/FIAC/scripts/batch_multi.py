import os

for design in ['event', 'block']:
    for which in ['contrasts', 'delays']:
        for contrast in ['average', 'speaker', 'sentence', 'interaction']:
            cmd = "python multi_run.py %s %s %s" % (design, which, contrast)
            os.system(cmd)


