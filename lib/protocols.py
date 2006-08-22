"""
Very lightweight system for checking protocol implementation on Python objects.
"""
from types import ClassType

class ProtocolError (Exception):
    "Indicate that a value does not support part of its expected protocol."

class Protocol: pass

class Iterable (Protocol):
    def __iter__(self): pass

class Iterator (Protocol):
    def next(self): pass

class Sequence (Iterable):
    def __len__(self): pass
    def __getitem__(self, index): pass
    def __getslice__(self, *args): pass

class MutableSequence (Sequence):
    def __setitem__(self, index, value): pass
    def __delitem__(self, index): pass

def haslength(obj):
    try:
        len(obj)
        return True
    except: return False

def protoset(obj):
    "@return the set of names representing the protocol supported by obj."
    proto = dir(obj)
    if type(obj) == ClassType and issubclass(obj, Protocol):
        proto.remove("__doc__")
        proto.remove("__module__")
    return set(proto)

def union(*sets):
    "@return the union of zero or more sets."
    if len(sets) == 0: return set()
    return sets[0].union(union(*sets[1:]))

def implements(value, *protocols):
    "@return True if value implements one of the given protocols."
    valproto = protoset(value)
    for proto in protocols:
        if protoset(proto).issubset(valproto): return True
    return False
