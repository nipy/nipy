import sys

class Needs:
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, *args, **kw):
        print ("Test not run. Requires %s flag"  % self.flag)

def _flag(func, flag):
    if flag not in sys.argv and "--all" not in sys.argv:
        return Needs(flag)
    else:
        return func

def slow(func):    
    return _flag(func, "--slow")

def gui(func):    
    return _flag(func, "--gui")

def data(func):    
    flag = "--data"
    if flag not in sys.argv and "--all" not in sys.argv:
        return Needs(flag)
    else:
        def _f(self):
            self.data_setUp()
            return func(self)
        return _f




@slow
def foo(x, y, z):
    print "foo"

if __name__ == '__main__':
    foo(1, 2, 3)
