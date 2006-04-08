from constants import *
from header import NIFTI_header_dict as _header
import struct, sys

# simplest validators simply check the packstr

NIFTI_validators = {}

_byteorder_dict = {'big':'>', 'little':'<'}

class Validator:
    def __init__(self, name, istuple, packstr):
        self.name = name
        self.istuple = istuple
        self.packstr = packstr

    def __call__(self, value, bytesign=_byteorder_dict[sys.byteorder]):
        if not self.istuple:
            packvalue = (value,)
        else:
            packvalue = tuple(value)
        try:
            apply(struct.pack, (self.packstr,) + packvalue)
        except:
            raise ValueError, 'NIFTI attribute ' + self.name + ':' + `value` + ' is misformatted, it should of type ' + `self.packstr`
        if not self.istuple:
            return packvalue[0]
        else:
            return packvalue
        

for att in _header.keys():
    packstr, istuple, default = _header[att]
    NIFTI_validators[att] = Validator(att, istuple, packstr)

# extra validators for specific fields

EXTRA_validators = {}

# datatype code

def validate_datatype(datatype):
    if datatype not in DT:
        raise ValueError, 'datatype:%s must have one of the following values: %s, see nifti1.h for details' % (`datatype`, `DT`)
    return True

EXTRA_validators['datatype'] = validate_datatype

# intent code

def validate_intent(intent):
    if intent not in NIFTI_INTENT:
        raise ValueError, 'intent_code:%s must have one of the following values: %s, see nifti1.h for details' % (`intent`, `NIFTI_INTENT`)
    return True

EXTRA_validators['intent_code'] = validate_intent

# slice code

def validate_slice(slice):
    if slice not in NIFTI_SLICE:
        raise ValueError, 'slice_code:%s must have one of the following values: %s, see nifti1.h for details' % (`slice`, `NIFTI_SLICE`)
    return True

EXTRA_validators['slice_code'] = validate_slice

for att in EXTRA_validators.keys():
    _default = NIFTI_validators[att]
    _extra = EXTRA_validators[att]
    def _validate(value, bytesign=_byteorder_dict[sys.byteorder]):
        if _extra(value):
            return _default(value, bytesign=bytesign)
    NIFTI_validators[att] = _validate # rewrite the default validator

