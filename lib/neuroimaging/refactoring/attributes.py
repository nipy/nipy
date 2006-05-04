
##############################################################################
class attribute (property):
    _attvals_name = "__attribute_values__"
    makeclass = False

    #-------------------------------------------------------------------------
    class __metaclass__ (type):
        def __new__(metaclass, name, bases, dictionary, makeclass=False):
            # return the attribute class
            if bases == (property,) or dictionary.get("makeclass") or makeclass:
                return type.__new__(metaclass, name, bases, dictionary)
            # return a attribute instance
            return metaclass(name, bases, dictionary, makeclass=True)(
              name, default=dictionary.get('default'),
              doc=dictionary.get('__doc__'))

    #-------------------------------------------------------------------------
    def __init__(self, name, default=None, doc=None):
        self.name = name
        self.default = default
        property.__init__(self,
          fget=self.get, fset=self.set, fdel=self.delete, doc=doc)

    #-------------------------------------------------------------------------
    def super(self): return super(self.__class__.__bases__[0], self)

    #-------------------------------------------------------------------------
    def isinitialized(self, host):
        return self._get_attvals(host).has_key(self.name)

    #-------------------------------------------------------------------------
    def _get_attvals(self, host):
        if not hasattr(host, self._attvals_name):
            setattr(host, self._attvals_name, {})
        return getattr(host, self._attvals_name)

    #-------------------------------------------------------------------------
    def get(self, host):
        attvals = self._get_attvals(host)
        if not attvals.has_key(self.name):
            if self.default is not None: attvals[self.name] = self.default
            else: raise AttributeError(self.name)
        return attvals[self.name]

    #-------------------------------------------------------------------------
    def set(self, host, value): self._get_attvals(host)[self.name] = value
        
    #-------------------------------------------------------------------------
    def delete(self, host):
        attvals = self._get_attvals(host)
        if attvals.has_key(self.name): del attvals[self.name]

class readonly (attribute):
    "A attribute which is read-only."
    makeclass = True
    def set(self, host, value):
        raise AttributeError("attribute %s is read only"%self.name)
 
class setonce (attribute):
    makeclass = True
    def set(self, host, value):
        if self.isinitialized(host):
            raise AttributeError(
              "setonce attribute %s has already been set"%self.name)
        else: self.super().set(host, value)

def _test():
    class Foo (object):
        class x (attribute):
            "test attribute x"; default=11
            def get(self, host):
                print "Customised getter: getting",self.name,"from",host
                return attribute.get(self, host)

        class y (setonce): "test attribute y";

    f = Foo()
    print f.x
    f.x = 10
    print f.x
    f.y = "fnorb"
    print f.y
    f.y = "fnarb"
    print f.y

if __name__ == "__main__": _test()

