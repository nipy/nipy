import urllib2, os

#data_path = "/home/analysis/FIAC"
#data_path = "/home/timl/src/nipy-data/fmri/FIAC"
data_path = "/home/timl/data/kff.stanford.edu/FIAC"
web_path = "http://kff.stanford.edu/FIAC"

def urlexists(url):
    if url[0:4] == 'http':
        try:
            test = urllib2.urlopen(url)
            del(test)
            return True
        except:
            return False
    else:
        return os.path.exists(url)
