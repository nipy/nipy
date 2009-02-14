__docformat__ = 'restructuredtext'



# class ExperimentalFactor(ExperimentalRegressor, Factor):
#     """
#     Return a factor that is a function of experimental time based on
#     an iterator. If the delta attribute is False, it is assumed that
#     the iterator returns rows of the form:

#     type, start, stop

#     Here, type is a hashable object and start and stop are floats.

#     If delta is True, then the events are assumed to be delta functions
#     and the rows are assumed to be of the form:

#     type, start

#     where the events are (square wave) approximations
#     of a delta function, non zero on [start, start+dt). 

#     Notes
#     -----

#     self[key] returns the __UNCONVOLVED__ factor, even if the
#     ExperimentalFactor has been convolved with an HRF. 
    

#     """
    
#     def __init__(self, name, iterator, convolved=False, delta=True, dt=0.02):
#         """
#         :Parameters:
#             name : TODO
#                 TODO
#             iterator : TODO
#                 TODO
#             convolved : bool
#                 TODO
#             delta : bool
#                 TODO
#             dt : float
#                 TODO
#         """
#         ExperimentalRegressor.__init__(self, convolved)
#         self.delta = delta
#         self.dt = dt
        
#         self.fromiterator(iterator)
#         keys = self.events.keys() + [downtime]
#         Factor.__init__(self, name, keys)
#         self._event_keys = self.events.keys()
#         self._event_keys.sort()
#         namespace[name] = self

#     def main_effect(self):
#         """
#         Return the 'main effect' for an ExperimentalFactor.
        
#         :Returns: `ExperimentalQuantitative`
#         """

#         _c = self.convolved
#         self.convolved = False
#         f = lambda t: f(t)
#         self.convolved = _c
#         return ExperimentalQuantitative('%s:maineffect' % self.termname, f)

#     def __getitem__(self, key):
#         """
#         :Parameters:
#             key : TODO
#                 TODO
        
#         :Returns: TODO
#         """

#         if self.events.has_key(key): # not in self.events.keys():
#                                      # this statement above seems useless 
#             l = self.events.keys()
#             l.sort()                 # sort the keys so output
#                                      # is consistent -- JT
#             j = l.index(key)
#         else:
#             raise KeyError, 'key not found'            

#         def factor_func(time, namespace=namespace, j=j,
#                 obj=self, **ignored):
#             _c = obj.convolved
#             obj.convolved = False
#             v = obj(time, namespace=namespace)[j]
#             obj.convolved = _c
#             return [np.squeeze(v) * 1.]

#         name = '%s[%s]' % (self.termname, `key`)
#         return ExperimentalQuantitative(name, factor_func)

#     def __call__(self, time, includedown=False, convolved=None, **kw):
#         """
#         :Parameters:
#             time : TODO
#                 TODO
#             includedown : ``bool``
#                 TODO
#             convolved : TODO
#                 TODO
#             kw : ``dict``
#                 TODO
        
#         :Returns: TODO
#         """
#         if convolved is not None:
#             __convolved, self.convolved = self.convolved, convolved
#         else:
#             __convolved = self.convolved

#         if not self.convolved:
#             value = []
#             keys = self.events.keys()
#             keys.sort()
#             for level in keys:
#                 value.append(np.squeeze(self.events[level](time)))
#             if includedown:
#                 s = np.add.reduce(value)

#                 keys = keys + [downtime]
#                 which = np.argmax(value, axis=0)
#                 which = np.where(s, which, keys.index(downtime))
#                 tmp, self.namespace = self.namespace, {self.termname:[keys[w] for w in which]}
#                 value = Factor.__call__(self)
#                 self.namespace = tmp
#             else:
#                 value = np.asarray(value)
#         else:
#             if hasattr(self, '_convolved'):
#                 value = self._convolved(time, **kw)
#             else:
#                 raise ValueError, 'no IRF defined for factor %s' % self.name
#         self.convolved = __convolved
#         return value

#     def names(self, keep=False):
#         """
#         :Parameters:
#             keep : bool
#                 TODO
        
#         :Returns: TODO
#         """
#         names = Factor.names(self)

#         _keep = []
#         for i, name in enumerate(names):
#             if name.find(downtime) < 0:
#                 _keep.append(i)

#         if not keep:
#             return [names[i] for i in _keep]
#         else:
#             return [names[i] for i in _keep], _keep

#     def fromiterator(self, iterator, delimiter=','):
#         """
#         Determine an ExperimentalFactor from an iterator
        
#         :Parameters:
#             iterator : TODO
#                 TODO
#             delimiter : string
#                 TODO
                
#         :Returns: ``None``
#         """

#         if isinstance(iterator, str):
#             iterator = csv.reader(file(iterator), delimiter=delimiter)
#         elif isinstance(iterator, file):
#             iterator = csv.reader(iterator, delimiter=delimiter)


