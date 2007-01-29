import os

for i in range(16):
    for run in range(1, 5):
        os.system("python run.py %d %d" % (i, run))
