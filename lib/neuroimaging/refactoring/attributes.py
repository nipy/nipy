from copy import copy
from sets import Set

class ProtocolOmission (Exception): pass

##############################################################################
class attribute (property):
    _attvals_name = "__attribute_values__"
    classdef = False
    valtype = None
    default = None
    readonly = False
    doc = ""
    implements = ()

    #-------------------------------------------------------------------------
    class __metaclass__ (type):
        def __new__(metaclass, classname, bases, classdict, classdef=False):
            # return the attribute class
            if bases == (property,) or classdict.get("classdef") or classdef:
                return type.__new__(metaclass, classname, bases, classdict)
            # return a attribute instance
            return metaclass(classname, bases, classdict, classdef=True)(
              classname, doc=classdict.get("__doc__"))

    #-------------------------------------------------------------------------
    @staticmethod
    def clone(att, **kwargs):
        """
        Static factory method for constructing a new attribute from an
        existing one, with modifications passed as keyword arguments.
        """
        newatt = copy(att)
        newatt.__dict__.update(kwargs)
        return newatt

    #-------------------------------------------------------------------------
    def __init__(self,
      name, valtype=None, default=None, readonly=None, doc=None):
        for argname in ("valtype","default","readonly"):
            argval = locals()[argname]
            if argval is not None: setattr(self, argname, argval)
        self.name = name
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
        defined = Set(dir(value))
        for protocol in self.implements:
            required = Set(dir(protocol))
            if not required.issubset(defined):
                raise ProtocolOmission, list(required - defined)

    #-------------------------------------------------------------------------
    def isvalid(self, host, value):
        try:
            self.validate(host, value)
            return True
        except TypeError, ProtocolOmission:
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
        if self.readonly and self.isinitialized(host):
            raise AttributeError(
              "attribute %s is read-only has already been set"%self.name)
        self.validate(host, value)
        self._get_attvals(host)[self.name] = value
        
    #-------------------------------------------------------------------------
    def delete(self, host):
        attvals = self._get_attvals(host)
        if attvals.has_key(self.name): del attvals[self.name]


##############################################################################
class wrapper (attribute):
    classdef=True
    attname=None
    def __init__(self, name, delegate, attname=None, readonly=None):
        attribute.__init(self, name)
        self.delegate = delegate,
        self.attname = self.attname or attname or name
        if readonly is not None: self.readonly = readonly
    def get(self, host):
        delegate = getattr(host,self.delegate.name)
        return getattr(delegate, self.attname or self.name)
    def set(self, host, value):
        if self.readonly:
            raise AttributeError("proxy %s is read-only"%self.name)
        delegate = getattr(host,self.delegate.name)
        setattr(delegate, self.attname or self.name, value)

def deferto(delegate):
    # note, this won't pull in delegate superclass stuff...
    framedict(1).update(
      dict([(attname,wrapper(attname,delegate))\
            for attname in delegate.__dict__.keys()]))

##############################################################################
class readonly (attribute):
    "A attribute which cannot be changed after it is initialized."
    classdef = True
    readonly = True
 
def _test():
    class Foo (object):
        class x (attribute):
            "test attribute x"; valtype=str; default=11
            def get(self, host):
                print "Customised getter: getting",self.name,"from",host
                return attribute.get(self, host)
        class y (readonly): "test attribute y"
        class z (readonly): "test attribute z"; default=10

    f = Foo()
    print f.x
    f.x = "borp"
    print f.x
    f.y = "fnorb"
    print f.y
    f.y = "fnarb"
    print f.y

if __name__ == "__main__": _test()

