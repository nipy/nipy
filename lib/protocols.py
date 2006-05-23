"""
Very lightweight system for checking protocol implementation on Python objects.
"""
from types import ClassType

class Protocol: pass
        

def protocol(*objects):
    """
    @return the tuple of names representing the complete protocol
    supported by the given objects.
    """
    if len(objects) == 0: return set()
    obj = objects[0]
    proto = set(dir(obj))
    if type(obj) == ClassType and issubclass(obj, Protocol):
        proto.discard("__doc__")
        proto.discard("__module__")
    return proto.union(protocol(*objects[1:]))

def implements(proto, value): return proto.issubset(protocol(value))

class ProtocolOmission (Exception):
    "Indicate that a value does not support part of its expected protocol."

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
