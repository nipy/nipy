"""
The attributes module provides a simple yet flexible attribute framework
for python classes, built on top of the built-in python properties.

>>> ###############################
>>> # Attribute Basics
>>> ###############################
>>>
>>> #------ typed attributes ------
>>>
>>> class PhotoIdentification (object):
...     photo=None
...
>>> class Wallet (object):
...     class dollars (attribute): default=0
...     class photoid (attribute): implements=PhotoIdentification
...
>>> w = Wallet()
>>> w.dollars
0
>>> w.dollars = 10
>>> w.dollars
10
>>> try: w.dollars = "gazillion"
... # can't set numeric attribute to a string
... except ProtocolOmission: print "can't do that" 
...
can't do that
>>> try: w.photoid = "not a photo id"
... except ProtocolOmission: print "photoid must have a photo"
...
photoid must have a photo
>>> class FakeID (object):
...     photo = "older brother"
...
>>> # photoid doesn't have to subclass PhotoIdentification, but it must
>>> # support the protocol (ie, must have an attribute called photo).
>>> w.photoid = FakeID()
>>> 
>>> #----- readonly attributes ----
>>> 
>>> class Newborn (object):
...    class name (readonly): pass
...
>>> n = Newborn()
>>> n.name = "Moonbeam"
>>> # readonly attributes can only be set once, so get it right the first time!
>>> try:
...     n.name = "Jenny"
... except AttributeError: print n.name
Moonbeam
>>> 
>>> #----- private attributes -----
>>>
>>> class FederalOfficial (object):
...     class name (readonly): default="John Doe"
...     def tell(self): return "secrets revealed"
...
... # Journalist reports what his source tells him.  No one can access the
... # source except the Journalist himself.
>>> class Journalist (object):
...     # _source is private because it starts with _
...     class _source (attribute): default=FederalOfficial()
...     class isemployed (attribute): default=True
...     def report(self): print self._source.tell() # self can access _source
...
>>> r = Journalist()
>>> r.report()
secrets revealed
>>> # try to find out the source's name
>>> try: print r._source.name
... # we can't access the source, so sack the journalist!
... except AccessError: r.isemployed = False
...
>>> r.isemployed
False
>>>
>>> ###############################
>>> # Delegation
>>> ###############################
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
    "Indicate that a value does not support part of its expected protocol."

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
    def _access_ok(self, host):
        return not self.isprivate() or scope(2).get("self") in (self, host)        

    #-------------------------------------------------------------------------
    def get(self, host):
        "Return attribute value on host."
        if not self._access_ok(host):
            raise AccessError("cannot get private attribute %s"%self.name)
        attvals = self._get_attvals(host)
        if not self.isinitialized(host): self.init(host)
        return attvals[self.name]

    #-------------------------------------------------------------------------
    def set(self, host, value):
        "Set attribute value on host."
        if not self._access_ok(host):
            raise AccessError("cannot set private attribute %s"%self.name)
        if self.readonly and self.isinitialized(host):
            raise AttributeError(
              "attribute %s is read-only but has already been set"%self.name)
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
        if not isinstance(delegate, attribute):
            raise ValueError("delegate must be an attribute")
        doc = "[Wrapper for %s.%s] "
        if delegate.__doc__: doc = doc + delegate.__doc__
        attribute.__init__(self, name, doc=doc)
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
        raise ValueError("please use only include or exclude but not both")
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


##############################################################################
class objectify (object):
    "Access a dictionary's key values like object attributes."
    class _dict (readonly): default={}
    deferto(_dict)

    def __init__(self, _dict=None):
        if _dict is not None: self._dict = _dict
    def __getattr__(self, name):
        if not self._dict.has_key(name): raise AttributeError(name)
        return self._dict[name]
    def __setattr__(self, name, value): self._dict[name] = value

# NOTE: objectify not working yet
def foo(): print objectify({'x':1,'y':2})
#foo()

#-----------------------------------------------------------------------------
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

