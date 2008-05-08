"""
odict.py - An object that provides an ordered dictionary.
"""
from copy import copy as _copy

class odict(dict):
    """
    This dictionary class extends dict to record the order in which items
    are added.  Calling keys(), values(), items(), etc. will return results in
    this order.
    """
    _keys = []

    def __init__(self, items=()):
        self._keys = map(lambda t: t[0], items)
        dict.__init__(self, items)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._keys.remove(key)

    def __setitem__(self, key, item):
        dict.__setitem__(self, key, item)
        if key not in self._keys: self._keys.append(key)

    def clear(self):
        dict.clear(self)
        self._keys = []

    def copy(self):
        """Copy this ordered dictionary.

        Returns a shallow copy of this dictionary.

        Examples
        --------
        
        >>> from neuroimaging.utils.odict import odict
        >>> origdict = odict((('one', 1.0), ('two', 2.0)))
        >>> origdict
        {'two': 2.0, 'one': 1.0}
        >>> newdict = origdict.copy()
        >>> newdict == origdict
        True
        >>> newdict is not origdict
        True

        """
        
        return _copy(self)

    def sort( self, keyfunc=None ):
        if keyfunc is None: self._keys.sort()
        else:
            decorated = [(keyfunc(key),key) for key in self._keys]
            decorated.sort()
            self._keys[:] = [t[1] for t in decorated]

    def items(self):
        return zip(self._keys, self.values())

    def keys(self):
        return self._keys

    def popitem(self):
        try:
            key = self._keys[-1]
        except IndexError:
            raise KeyError('dictionary is empty')

        val = self[key]
        del self[key]
        return (key, val)

    def setdefault(self, key, failobj = None):
        if key not in self._keys: self._keys.append(key)
        return dict.setdefault(self, key, failobj)

    def update(self, other):
        dict.update(self, other)
        for key in other.keys():
            if key not in self._keys: self._keys.append(key)

    def values(self):
        return map(self.get, self._keys)
