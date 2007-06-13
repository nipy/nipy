try:
    from enthought.traits.api import TraitHandler, Trait, HasTraits, Dict
    from enthought.traits.trait_errors import TraitError

except ImportError:
    from neuroimaging.utils.enthought.traits import TraitHandler, Trait, HasTraits, Dict
    from neuroimaging.utils.enthought.traits.trait_errors import TraitError

##############################################################################

class HasReadOnlyTraits(HasTraits):
    readonly_set = Dict


class ReadOnlyValidateHandler(TraitHandler):

    '''
    A set-once, read only trait with validation based on the trait
    passed in the construction.
    '''

    def __init__(self, trait):
        self.trait = Trait(trait)

    def validate(self, object, name, value):
        if hasattr(object, 'readonly_set'):
            if not object.readonly_set.has_key(name):
                object.readonly_set[name] = 1
            cur = getattr(object, name)

            if cur != value:
                object.readonly_set[name] += 1

            if object.readonly_set[name] <= 2:
                return self.trait.validate(object, name, value)
            else:
                self.error(object, name, value)
        else:
            return self.trait.validate(object, name, value)
            
    def error(self, object, name, value):
        raise TraitError, 'cannot modify read-only trait "%s"' % name

def ReadOnlyValidate(trait, **keywords):
    validator = ReadOnlyValidateHandler(trait)
    return Trait(trait, validator, **keywords)
