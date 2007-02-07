import os

for i in range(16):
    for run in range(1, 5):
##        os.system("python model.py %d %d" % (i, run))
        os.system("python compare.py %d %d" % (i, run))
