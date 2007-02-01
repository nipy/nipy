import urllib2, os

data_path = "/home/analysis/FIAC"

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
