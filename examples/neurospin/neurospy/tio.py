from numpy import *
import numpy

textypes = ['ascii','binar']

byteordertypes = {'DCBA':'bigindian',
                  'ABCD':'littleindian'}

datatypes = {'U8':uint8,
             'S8':int8,
             'U16':uint16,
             'S16':int16,
             'U32':uint32,
             'S32':int32,
             'FLOAT': float32,
             'DOUBLE':float64,
             'CFLOAT':clongfloat,
             'CDOUBLE':clongdouble
             }

class Texture:
    def __init__(self, filename, textype='ascii', byteorder='bigindian',
                 datatypesize=5, datatype=float32, nb_t=1, data=array([])):
        # TODO : virer le datatypesize
        # TODO : faire le datatype automatique en fonction du type dans data
        self.filename     = filename
        self.textype      = textype
        self.byteorder    = byteorder
        self.datatypesize = uint32(datatypesize)
        self.datatype     = datatype
        self.nb_t         = uint32(nb_t)
        self.nbitems      = uint32(len(data))
        self.data         = data

    def show(self):
        print 'textype:',self.textype
        print 'byteorder:',self.byteorder
        print 'datatypesize:',self.datatypesize
        print 'datatype:',self.datatype
        print 'nb_t:',self.nb_t
        print 'nbitems:',self.nbitems

    def copy(self):
        return Texture(self.filename, self.textype, self.byteorder,
                       self.datatypesize, self.datatype, self.nb_t,
                       array(self.data))

    def convertToBinary(self):
        self.datatypesize = uint32(self.datatypesize)
        self.nb_t = uint32(self.nb_t)
        


class TextureInputOutput:
    # appeler p.data pour le tableau de donnees

    @staticmethod
    def read(file):
        p = Texture(file)

        f_in = open(p.filename)

        p.textype = f_in.read(5)
        if p.textype not in textypes:
            raise TypeError, 'On char 0: Texture type not regular, it should\
            be \'ascii\' or \'binar\''

        if p.textype == textypes[1]: # BINARY

            p.byteorder = byteordertypes.get(f_in.read(4))
            if p.byteorder == None:
                raise TypeError, 'On char 5: data type not understood, it has\
                to be ABCD for littleindian and DCBA for bigindian'
            if p.byteorder == byteordertypes.get('ABCD'):
                raise TypeError, 'On char 5: Littleindian byteorder not\
                supported yet.'
            
            p.datatypesize = (frombuffer(f_in.read(4),uint32))[0]
            p.datatype     = datatypes[f_in.read(p.datatypesize)]

            if p.datatype=='_':
                raise TypeError, 'Datatype not supported yet.'
            
            p.nb_t    = (frombuffer(f_in.read(4),uint32))[0]

            # TODO some sanity check on data length
            p.data = []
            for t in range(p.nb_t):
                current_t = (frombuffer(f_in.read(4),uint32))[0]
                p.nbitems = (frombuffer(f_in.read(4),uint32))[0]
                p.data.append(frombuffer(f_in.read(p.nbitems*p.datatype().nbytes),p.datatype))

        else: # ASCII

            # on vire le \n de la fin de la premiere ligne
            f_in.readline()

            p.byteorder = byteordertypes.get('DCBA') # par defaut

            datatypetemp = f_in.readline()

            p.datatype = datatypes[datatypetemp[:-1]] # on vire le \n a la fin
            p.datatypesize = len([k for k, v in datatypes.items() if v == p.datatype][0])

            nb_t_temp = f_in.readline()
            p.nb_t = int(nb_t_temp[0:len(nb_t_temp)]) # on vire le\n


            datatemp = fromstring(string=f_in.readline(),
                                  sep=' ', dtype=p.datatype)

            p.data = []
            pos = 0
            for t in range(p.nb_t):
                current_t = int(datatemp[pos])
                pos+=1
                p.nbitems = int(datatemp[pos])
                pos+=1
                p.data.append(array(datatemp[pos:pos+p.nbitems]))
                pos+=p.nbitems

                
        p.data = array(p.data)
        f_in.close()
        return p


    @staticmethod
    def write(texture,filename=""):

        if filename=="":
            filename = texture.filename
            
        f_out = open(filename, 'w')

        # si ascii :
        if texture.textype == textypes[0]:
            f_out.write(texture.textype+'\n')
            f_out.write([k for k, v in datatypes.items() if v == texture.datatype][0]+'\n')
            f_out.write(str(texture.nb_t)+'\n')

            # ecrit les donnees en gerant la dimension t
            if texture.nb_t>1 :
                for t in range(texture.nb_t):
                    e = texture.data[t]
                    f_out.write(str(t)+' '+str(len(e))+' ')
                    e.tofile(f_out, sep=' ')
                    f_out.write(' ')
            else:
                f_out.write('0 '+str(texture.nbitems)+' ')
                texture.data.tofile(f_out, sep=' ')
                f_out.write(' ')
                



        # si binaire
        else:
            texture.convertToBinary()
            f_out.write(texture.textype)
            f_out.write([k for k, v in byteordertypes.items()
                         if v == texture.byteorder][0])
            f_out.write(texture.datatypesize.tostring())
            f_out.write([k for k, v in datatypes.items()
                         if v == texture.datatype][0])
            f_out.write(texture.nb_t.tostring())
            
            # ecrit les donnees en gerant la dimension t
            if texture.nb_t==1 :
                f_out.write('\x00\x00\x00\x00')
                f_out.write(uint32(len(texture.data)).tostring())
                f_out.write(texture.data.tostring())
            else:
                for t in range(texture.nb_t):
                    f_out.write(uint32(t).tostring())
                    e = texture.data[t]                   
                    f_out.write(uint32(len(e)).tostring())
                    f_out.write(e.tostring())
                    
