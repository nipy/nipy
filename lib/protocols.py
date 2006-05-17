from sets import Set

class ProtocolOmission (Exception):
    "Indicate that a value does not support part of its expected protocol."

def protocol(*objects):
    """
    @return the tuple of names representing the complete protocol
    supported by the given objects.
    """
    protocol = Set()
    for obj in objects:
        protocol = protocol.union(Set([name for name in dir(obj)]))
    return protocol

def implements(proto, value): return proto.issubset(protocol(value))

class Iterable:
    def __iter__(self): pass

class Sequence (Iterable):
    def __len__(self): pass
    def __getitem__(self, index): pass
    def __getslice__(self, *args): pass

class MutableSequence (Sequence):
    def __setitem__(self, index, value): pass
    def __setslice__(self, *args): pass
