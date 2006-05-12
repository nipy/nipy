"""
>>> #-----------------------------
>>> # Using basic attributes
>>> #-----------------------------
>>> #-----------------------------
>>> # Using wrappers and deferto
>>> #-----------------------------
>>>
>>> class Stomach (object):
...     acid = "hydrochloric acid"
...     def speak(self): print "I'm hungry!"
...
>>> class Mouth (object):
...     numteeth=28
...     def speak(self): print "blahblahblah"
...
>>> class Brain (object):
...     def speak(self): print "hmmm"
...
>>> class Head (object):
...     "Flaps at the jaw"
...     class stomach (attribute): default=Stomach()
...     class mouth (attribute): default=Mouth()
...     class brain (attribute): default=Brain()
...     deferto(mouth)
...
>>> h = Head()
>>> h.numteeth
28
>>> h.speak()
blahblahblah
>>> class ThoughtfulHead (Head):
...     "Speaks its mind"
...     deferto(Head.brain)
...
>>> h = ThoughtfulHead()
>>> h.speak()
hmmm
>>> h.numteeth
28
>>> class HungryHead (Head):
...     "Lets the stomach do the talking, but no acid reflux"
...     deferto(Head.stomach, "speak")
...
>>> h = HungryHead()
>>> h.speak()
I'm hungry!
>>> h.acid
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
AttributeError: 'HungryHead' object has no attribute 'acid'
"""
from sys import _getframe as getframe
from copy import copy
from sets import Set
from types import TupleType, ListType

class AccessError (Exception):
    "Indicate that a private attribute was referred to outside its class."

class ProtocolOmission (Exception):
    "Indicate that a value does not support part of its intended protocol."

def protocol(something):
    """
    @return the tuple of names representing the protocol supported by the
    given object.
    """
    return tuple([name for name in dir(something) if name[0] != "_"])

def scope(num): return getframe(num+1).f_locals


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

        # if no protocol is specified, use protocol of the default value
        if len(self.implements)==0 and self.default is not None:
            self.implements = (self.default,)

        property.__init__(self,
          fget=self.get, fset=self.set, fdel=self.delete, doc=doc)

    #-------------------------------------------------------------------------
    def _get_attvals(self, host):
        if not hasattr(host, self._attvals_name):
            setattr(host, self._attvals_name, {})
        return getattr(host, self._attvals_name)

    #-------------------------------------------------------------------------
    def super(self, base=None):
        return super(base or self.__class__.__bases__[0], self)

    #-------------------------------------------------------------------------
    def isinitialized(self, host):
        return self._get_attvals(host).has_key(self.name)

    #-------------------------------------------------------------------------
    def validate(self, host, value):
        "Make sure the value satisfies any implemented protocols"
        defined = Set(dir(value))
        for protocol in self.implements:
            required = Set(dir(protocol))
            if not required.issubset(defined): raise ProtocolOmission(
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
    def isprivate(self): return self.name[0] == "_"

    #-------------------------------------------------------------------------
    def init(self, host):
        "Called when attribute value is requested but has not been set yet."
        if self.default is not None: self.set(host, self.default)
        else: raise AttributeError("attribute %s is not initialized"%self.name)

    #-------------------------------------------------------------------------
    def get(self, host):
        "Return attribute value on host."
        if self.isprivate() and scope(1).get("self") != host:        
            raise AccessError("cannot get private attribute %s"%self.name)
        attvals = self._get_attvals(host)
        if not self.isinitialized(host): self.init(host)
        return attvals[self.name]

    #-------------------------------------------------------------------------
    def set(self, host, value):
        "Set attribute value on host."
        if self.isprivate() and scope(1).get("self") != host:        
            raise AccessError("cannot set private attribute %s"%self.name)
        if self.readonly and self.isinitialized(host):
            raise AttributeError(
              "attribute %s is read-only has already been set"%self.name)
        self.validate(host, value)
        if len(self.implements)==0: self.implements = (value,)
        self._get_attvals(host)[self.name] = value
        
    #-------------------------------------------------------------------------
    def delete(self, host):
        "Delete attribute from host (attribute will be uninitialized)."
        attvals = self._get_attvals(host)
        if attvals.has_key(self.name): del attvals[self.name]


##############################################################################
class wrapper (attribute):
    "Wrap an attribute or method of another attribute."
    classdef=True
    attname=None
    def __init__(self, name, delegate, attname=None, readonly=None):
        attribute.__init__(self, name)
        self.delegate = delegate
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
def deferto(delegate, include=(), exclude=()):
    if include and exclude:
        raise ValueError("please use only include or exclude")
    if not isinstance(delegate, attribute):
        raise ValueError("delegate must be an attribute")
    scope(1).update(
      dict([(name,wrapper(name,delegate))\
            for name in delegate.protocol\
            if (not include or name in include) and\
               (not exclude or name not in exclude)]))


##############################################################################
class readonly (attribute):
    "A attribute which cannot be changed after it is initialized."
    classdef = True
    readonly = True
 
def _test():
    class Foo (object):
        class x (attribute):
            "test attribute x"; implements=str; default="foo"
            def get(self, host):
                print "Customised getter: getting",self.name,"from",host
                return attribute.get(self, host)
        class y (readonly): "test attribute y"
        class z (readonly): "test attribute z"; default=10
        class _a (attribute): "private attribute"

        def get_a(self): return self._a
        def set_a(self, value): self._a = value

    f = Foo()
    print f.x
    f.x = "borp"
    print f.x
    f.y = "fnorb"
    print f.y
    #f.y = "fnarb"
    #print f.y
    f.set_a(10)
    print f.get_a()
    print f._a
    f._a = 10

if __name__ == "__main__":
    from doctest import testmod
    testmod()

