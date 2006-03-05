from struct import *
import enthought.traits as traits

class BinaryHeaderValidator(traits.TraitHandler):

    def __init__(self, packstr, value=None, seek=0, bytesign = '>', **keywords):
        for key, value in keywords.items():
            setattr(self, key, value)
        self.seek = seek
        self.packstr = packstr
        self.bytesign = bytesign

    def write(self, value, outfile=None):

        try:
            if self.packstr[-1] != 's':
                packvalue = tuple(value)
            else:
                packvalue = (value,)
        except:
            packvalue = (value,)
            
        result = apply(pack, (self.bytesign + self.packstr,) + packvalue)
        if outfile is not None:
            outfile.seek(self.seek)
            outfile.write(result)
        return result

    def validate(self, object, name, value):

        try:
            result = self.write(value)
        except:
            self.error(object, name, value)

        _value = unpack(self.bytesign + self.packstr, result)

        if is_tupled(self.packstr, _value):
            return _value
        else:
            return _value[0]

    def info(self):
        return 'an object of type "%s", apply(struct.pack, "%s", object) must make sense' % (self.packstr, self.packstr)

    def read(self, hdrfile):
        hdrfile.seek(self.seek)
        value = unpack(self.bytesign + self.packstr,
                       hdrfile.read(calcsize(self.packstr)))
        if not is_tupled(self.packstr, value):
            value = value[0]
        return value

def is_tupled(packstr, value):
    try:
        if packstr[-1] != 's':
            packvalue = tuple(value)
            if len(packvalue) > 1:
                return True
            else:
                return False
        else:
            return False
    except:
        return False


def BinaryHeaderAtt(packstr, value=None, **keywords):
    validator = BinaryHeaderValidator(packstr, value=value, **keywords)
    return traits.Trait(value, validator)
        
