from sys import _getframe as getframe
from copy import copy
from sets import Set
from types import TupleType, ListType

class ProtocolOmission (Exception):
    "Indicate that a value does not support part of its intended protocol."

def protocol(something):
    """
    @return the tuple of names representing the protocol supported by the
    given object.
    """
    return tuple([name for name in dir(something) if name[0] != "_"])


##############################################################################
class attribute (property):
    _attvals_name = "__attribute_values__"
    classdef = False
    default = None
    readonly = False
    doc = ""
    implements = ()

    def _get_protocol(self):
        total_proto = Set()
        for proto in self.implements:
            total_proto = total_proto.union(Set(protocol(proto)))
        return total_proto
    protocol = property(_get_protocol)

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
      name, implements=None, default=None, readonly=None, doc=None):
        self.name = name

        # make sure implements is a sequence
        if implements is not None: self.implements = implements
        if type(self.implements) not in (TupleType, ListType):
            self.implements = (self.implements,)

        # use or override the class default for these
        for argname in ("default","readonly"):
            argval = locals()[argname]
            if argval is not None: setattr(self, argname, argval)

        if len(self.implements)==0 and default is not None:
            self.implements = (default,)

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
        "Make sure the value satisfies any implemented protocols"
        defined = Set(dir(value))
        for protocol in self.implements:
            required = Set(dir(protocol))
            if not required.issubset(defined):
                print "value=",value
                print "protocol=",protocol
                print "defined =",defined
                print "required =",required
                raise ProtocolOmission(
                  "attribute %s implements %s, value %s does not implement: %s"%\
                  (self.name,self.implements,value,tuple(required - defined)))

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
        attribute.__init__(self, name)
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

#-----------------------------------------------------------------------------
def deferto(delegate):
    if not isinstance(delegate, attribute):
        raise ValueError("delegate must be an attribute")
    getframe(1).f_locals.update(
      dict([(attname,wrapper(attname,delegate))\
            for attname in delegate.protocol]))


##############################################################################
class readonly (attribute):
    "A attribute which cannot be changed after it is initialized."
    classdef = True
    readonly = True
 
def _test():
    class Foo (object):
        class x (attribute):
            "test attribute x"; implements=str; default=11
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

