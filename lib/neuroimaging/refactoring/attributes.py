import sets

class InterfaceOmission (Exception): pass

##############################################################################
class attribute (property):
    _attvals_name = "__attribute_values__"
    makeclass = False
    valtype = None
    implements = ()

    #-------------------------------------------------------------------------
    class __metaclass__ (type):
        def __new__(metaclass, classname, bases, classdict, makeclass=False):
            # return the attribute class
            if bases == (property,) or classdict.get("makeclass") or makeclass:
                return type.__new__(metaclass, classname, bases, classdict)
            # return a attribute instance
            return metaclass(classname, bases, classdict, makeclass=True)(
              classname,
              valtype=classdict.get("valtype"),
              default=classdict.get("default"),
              doc=classdict.get("__doc__"))

    #-------------------------------------------------------------------------
    def __init__(self, name, valtype=None, default=None, doc=None):
        self.name = name
        self.valtype = valtype
        self.default = default
        if self.valtype is None and default is not None:
            self.valtype = type(default)
        property.__init__(self,
          fget=self.get, fset=self.set, fdel=self.delete, doc=doc)

    #-------------------------------------------------------------------------
    def super(self, base=None):
        return super(base or self.__class__.__bases__[0], self)

    #-------------------------------------------------------------------------
    def isinitialized(self, host):
        return self._get_attvals(host).has_key(self.name)

    #-------------------------------------------------------------------------
    def _get_attvals(self, host):
        if not hasattr(host, self._attvals_name):
            setattr(host, self._attvals_name, {})
        return getattr(host, self._attvals_name)

    #-------------------------------------------------------------------------
    def validate(self, host, value):
        # type check
        if self.valtype is not None and \
          not issubclass(type(value), self.valtype):
            raise TypeError,\
              "attribute %s value %s must have type %s"% \
              (self.name,`value`,self.valtype)

        # protocol check
        defined = sets.Set(dir(value))
        for protocol in self.implements:
            required = sets.Set(dir(protocol))
            if not required.issubset(defined):
                raise InterfaceOmission, list(required - defined)

    #-------------------------------------------------------------------------
    def isvalid(self, host, value):
        try:
            self.validate(host, value)
            return True
        except TypeError, InterfaceOmission:
            return False

    #-------------------------------------------------------------------------
    def get(self, host):
        attvals = self._get_attvals(host)
        if not attvals.has_key(self.name):
            if self.default is not None:
                self.set(host, self.default)
            else: raise AttributeError(self.name)
        return attvals[self.name]

    #-------------------------------------------------------------------------
    def set(self, host, value):
        self.validate(host, value)
        self._get_attvals(host)[self.name] = value
        
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
            "test attribute x"; valtype=str; default=11
            def get(self, host):
                print "Customised getter: getting",self.name,"from",host
                return attribute.get(self, host)

        class y (setonce): "test attribute y";

        class z (readonly): default=10

    f = Foo()
    print f.x
    f.x = "borp"
    print f.x
    f.y = "fnorb"
    print f.y
    f.y = "fnarb"
    print f.y

if __name__ == "__main__": _test()

