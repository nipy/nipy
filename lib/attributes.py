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
... except ProtocolError: print "can't do that" 
...
can't do that
>>> try: w.photoid = "not a photo id"
... except ProtocolError: print "photoid must have a photo"
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
>>>
>>> class ThoughtfulHead (Head):
...     "Speaks its mind"
...     deferto(Head.brain)
...
>>> h = ThoughtfulHead()
>>> h.speak()
hmmm
>>> h.numteeth
28
>>>
>>> class HungryHead (Head):
...     "Lets the stomach do the talking, but has no acid"
...     deferto(Head.stomach, ("speak",))
...
>>> h = HungryHead()
>>> h.speak()
I'm hungry!
>>> h.acid
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
AttributeError: 'HungryHead' object has no attribute 'acid'
>>>
"""
from sys import _getframe as getframe
from copy import copy
from types import TupleType, ListType

import protocols
from protocols import protoset, union, implements, ProtocolError, Sequence

class AccessError (Exception):
    "Indicate that a private attribute was referred to outside its class."

def scope(num): return getframe(num+1).f_locals

# TODO:
#  special UNSET value for default, to distinguish a legit value of None

##############################################################################
class attribute (property):
    _attvals_name = "__attribute_values__"
    classdef = False
    default = None
    readonly = False
    doc = None
    implements = (None,)

    #-------------------------------------------------------------------------
    class __metaclass__ (type):
        def __new__(metaclass, classname, bases, classdict, classdef=False):
            # return the attribute class
            if bases == (property,) or classdict.get("classdef") or classdef:
                return type.__new__(metaclass, classname, bases, classdict)
            # return a attribute instance
            return metaclass(classname, bases, classdict, classdef=True)(
              classname)

    #-------------------------------------------------------------------------
    def __init__(self,
      name, implements=None, default=None, readonly=None, doc=None):
        self.name = name

        # make sure implements is a sequence
        if implements is not None: self.implements = implements
        if not type(self.implements) == type(()):
            self.implements = (self.implements,)

        # use or override the class default for these
        for argname in ("default","readonly"):
            argval = locals()[argname]
            if argval is not None: setattr(self, argname, argval)

        # if no protocol is specified, use protocol of the default value
        if len(self.implements)==0 and self.default is not None:
            self.implements = (self.default,)

        self.doc = (doc is None) and self.__doc__ or doc
        property.__init__(self, fget=self.get, fset=self.set, fdel=self.delete)

    #-------------------------------------------------------------------------
    def _get_attvals(self, host):
        if not hasattr(host, self._attvals_name):
            setattr(host, self._attvals_name, {})
        return getattr(host, self._attvals_name)

    #-------------------------------------------------------------------------
    def clone(self, **kwargs):
        """
        Return a copy of self, with modifications passed as keyword arguments.
        """
        newatt = self.__class__(self.name)
        newatt.__dict__.update(kwargs)
        return newatt

    #-------------------------------------------------------------------------
    def super(self, base=None):
        return super(base or self.__class__.__bases__[0], self)

    #-------------------------------------------------------------------------
    def isinitialized(self, host):
        return self._get_attvals(host).has_key(self.name)

    #-------------------------------------------------------------------------
    def validate(self, value):
        "Raise an exception if the value is not valid."
        if not self.isvalid(value):
            raise ProtocolError("value %s of type %s must implement one of: %s"%\
              (value, type(value), self.implements))

    #-------------------------------------------------------------------------
    def isvalid(self, value):
        "Return whether the value satisfies all implemented protocols"
        return implements(value, *self.implements)

    #-------------------------------------------------------------------------
    def isprivate(self): return self.name[0] == "_"

    #-------------------------------------------------------------------------
    def _initialize(self, host):
        "Called when attribute value is requested and has not yet been set."
        value = self.init(host)
        if value is not None: self.set(host, value)
        else: raise AttributeError("attribute %s is not initialized on %s"%\
          (self.name, host))

    #-------------------------------------------------------------------------
    def init(self, host): return self.default

    #-------------------------------------------------------------------------
    def _access_ok(self, host):
        return not self.isprivate() or scope(2).get("self") in (self, host)        

    #-------------------------------------------------------------------------
    def get(self, host):
        "Return attribute value on host."
        if not self._access_ok(host):
            raise AccessError("cannot get private attribute %s"%self.name)
        attvals = self._get_attvals(host)
        if not self.isinitialized(host): self._initialize(host)
        return attvals[self.name]

    #-------------------------------------------------------------------------
    def set(self, host, value):
        "Set attribute value on host."
        if not self._access_ok(host):
            raise AccessError("cannot set private attribute %s"%self.name)
        if self.readonly and self.isinitialized(host):
            raise AttributeError(
              "attribute %s is read-only and has already been set"%self.name)
        self.validate(value)
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

    #-------------------------------------------------------------------------
    def __init__(self, name, delegate, attname=None, readonly=None, doc=None):
        if delegate is not None: self.delegate = delegate
        if not isinstance(self.delegate, attribute):
            raise ValueError("delegate must be an attribute")
        self.attname = self.attname or attname or name
        if doc is None:
            doc = "[Wrapper for %s.%s] "%(delegate.name, self.attname)
            if delegate.__doc__: doc = doc + delegate.__doc__
        attribute.__init__(self, name, doc=doc)
        if readonly is not None: self.readonly = readonly

    #-------------------------------------------------------------------------
    def _host_delegate(self, host):return getattr(host, self.delegate.name)

    #-------------------------------------------------------------------------
    def get(self, host):
        return getattr(self._host_delegate(host), self.attname)

    #-------------------------------------------------------------------------
    def set(self, host, value):
        if self.readonly:
            raise AttributeError("wrapper %s is read-only"%self.name)
        delegate = getattr(host,self.delegate.name)
        setattr(self._host_delegate(host), self.attname, value)

#-----------------------------------------------------------------------------
def deferto(delegate, include=(), exclude=(), privates=False):
    if include and exclude:
        raise ValueError("please use only include or exclude but not both")

    # include privates if a private is explicitly specified in the includes
    if filter(lambda n: n[0]=='_', include): privates = True

    # make sure the delegate supports the inclusions
    delegate_proto = union(*map(protoset, delegate.implements))
    includeset = set(include)
    if not includeset.issubset(delegate_proto):
        raise ValueError("delegate does not implement %s"%\
          tuple(includeset - delegate_proto))

    scope(1).update(
      dict([(name,wrapper(name,delegate))\
            for name in delegate_proto if (privates or name[0]!="_") and\
              (not include or name in includeset) and\
              (not exclude or name not in exclude)]))

#-----------------------------------------------------------------------------
def clone(att, **kwargs):
    """
    Add a clone of the given attribute to the calling scope.  The clone can
    be modified with the keyword args.
    """
    name = kwargs.get("name", att.name)
    scope(1)[name] = att.clone(**kwargs)


##############################################################################
class readonly (attribute):
    "An attribute which cannot be changed after it is initialized."
    classdef = True
    readonly = True


##############################################################################
class constant (attribute):
    "An attribute which can never be set (must specify a default value)."
    classdef = True
    def get(self, host): return self.default
    def set(self, host, value): raise AttributeError(
      "attribute %s is constant (value=%s) and cannot be set"%\
      (self.name, self.default))


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

if __name__ == "__main__":
    from doctest import testmod
    testmod()

