import numpy as N
import os

from neuroimaging.utils.odict import odict
import neuroimaging.data_io as dataio
from neuroimaging.data_io import DataSource
import neuroimaging.data_io.formats.binary as bin
from neuroimaging.data_io.formats import utils

from neuroimaging.core.reference.axis import space, spacetime
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.utils.path import path

# ECAT 7 header
HEADER_SIZE = 512
BLOCKSIZE = 512
SWVERSION = 72

# Matrix Data Types
ECAT7_BYTE = 1
ECAT7_VAXI2 = 2
ECAT7_VAXI4 = 3
ECAT7_VAXR4 = 4
ECAT7_IEEER4 = 5
ECAT7_SUNI2 = 6
ECAT7_SUNI4 = 7

# map ECAT datatype to numpy scalar type
datatype2sctype = {
    ECAT7_BYTE: N.uint8, 
    ECAT7_VAXI2: N.ushort,
    ECAT7_VAXI4: N.int16,
    ECAT7_VAXR4: N.float,
    ECAT7_IEEER4: N.float,
    ECAT7_SUNI2: N.ushort,
    ECAT7_SUNI4: N.int16}

sctype2datatype = dict([(k,v) for k,v in datatype2sctype.items()])


# Matrix File Types
ECAT7_UNKNOWN = 0
ECAT7_2DSCAN  = 1
ECAT7_IMAGE16 = 2
ECAT7_ATTEN   = 3
ECAT7_2DNORM  = 4
ECAT7_POLARMAP = 5
ECAT7_VOLUME8  = 6
ECAT7_VOLUME16 = 7
ECAT7_PROJ = 8
ECAT7_PROJ16 = 9
ECAT7_IMAGE8 = 10
ECAT7_3DSCAN = 11
ECAT7_3DSCAN8 = 12
ECAT7_3DNORM = 13
ECAT7_3DSCANFIT = 14

# Patient Orientation
ECAT7_Feet_First_Prone = 0
ECAT7_Head_First_Prone = 1
ECAT7_Feet_First_Supine = 2
ECAT7_Head_First_Supine = 3
ECAT7_Feet_First_Decubitus_Right = 4
ECAT7_Head_First_Decubitus_Right = 5
ECAT7_Feet_First_Decubitus_Left = 6
ECAT7_Head_First_Decubitus_Left = 7
ECAT7_Unknown_Orientation = 8

# ECAT 7 MAIN Header Fields
struct_formats_mh = odict((
    ('magic_number', '14s'),
    ('original_file_name', '32s'),
    ('sw_version', 'H'),
    ('system_type', 'H'),
    ('file_type', 'H'),
    ('serial_number', '10s'),
    ('scan_start_time', 'l'),
    ('isotope_name', '8s'),
    ('isotope_halflife', 'f'),
    ('radiopharmaceutical', '32s'),
    ('gantry_tilt', 'f'),
    ('gantry_rotation', 'f'),
    ('bed_elevation', 'f'),
    ('intrinsic_tilt', 'f'),           
    ('wobble_speed', 'H'),             
    ('transm_source_type', 'H'),
    ('distance_scanned', 'f'),
    ('transaxial_fov', 'f'),           
    ('angular_compression', 'H'),
    ('coin_samp_mode', 'H'),
    ('axial_samp_mode', 'H'),          
    ('ecat_calibration_factor','f'),
    ('calibration_units', 'H'),        
    ('calibration_units_type','H'),   
    ('compression_code', 'H'),             
    ('study_type', '12s'),              
    ('patient_id', '16s'),               
    ('patient_name', '32s'),             
    ('patient_sex', 's'),              
    ('patient_dexterity', 's'),        
    ('patient_age', 'f'),              
    ('patient_height', 'f'),           
    ('patient_weight', 'f'),           
    ('patient_birth_date', 'I'),       
    ('physician_name', '32s'),
    ('operator_name', '32s'),          
    ('study_description', '32s'),        
    ('acquisition_type', 'H'),        
    ('patient_orientation', 'H'),     
    ('facility_name', '20s'),            
    ('num_planes', 'H'),               
    ('num_frames', 'H'),               
    ('num_gates', 'H'),               
    ('num_bed_pos', 'H'),    
    ('init_bed_position', 'f'),
    ('bed_position(1)', 'f'),
    ('bed_position(2)', 'f'),
    ('bed_position(3)', 'f'),
    ('bed_position(4)', 'f'),  
    ('bed_position(5)', 'f'),  
    ('bed_position(6)', 'f'),  
    ('bed_position(7)', 'f'),  
    ('bed_position(8)', 'f'),  
    ('bed_position(9)', 'f'),  
    ('bed_position(10)', 'f'),  
    ('bed_position(11)', 'f'),  
    ('bed_position(12)', 'f'),  
    ('bed_position(13)', 'f'),  
    ('bed_position(14)', 'f'),  
    ('bed_position(15)', 'f'),  
    ('plane_separation', 'f'),
    ('lwr_sctr_thres', 'H'), 
    ('lwr_true_thres', 'H'),           
    ('upr_true_thres', 'H'),           
    ('user_process_code', '10s'),
    ('acquisition_mode', 'H'),
    ('bin_size', 'f'),                 
    ('branching_fraction', 'f'),       
    ('dose_start_time', 'f'),          
    ('dosage', 'f'),                   
    ('well_counter_corr_factor', 'f'), 
    ('data_units', '32s'), 
    ('septa_state', 'H'),   
    ('fill', 'H')
))
field_formats_mh = struct_formats_mh.values();

# ECAT 7 SUB Header Fields
struct_formats_sh = odict((
    ('DATA_TYPE', 'H'),
    ('NUM_DIMENSIONS','H'),
    ('X_DIMENSION','H'),
    ('Y_DIMENSION','H'),
    ('Z_DIMENSION','H'),
    ('X_OFFSET','f'),
    ('Y_OFFSET','f'),
    ('Z_OFFSET', 'f'),
    ('RECON_ZOOM','f'),
    ('SCALE_FACTOR','f'),
    ('IMAGE_MIN', 'h'),
    ('IMAGE_MAX', 'h'),
    ('X_PIXEL_SIZE', 'f'), 
    ('Y_PIXEL_SIZE', 'f'), 
    ('Z_PIXEL_SIZE', 'f'), 
    ('FRAME_DURATION', 'f'), 
    ('FRAME_START_TIME', 'f'),
    ('FILTER_CODE', 'H'),
    ('X_RESOLUTION','f'),
    ('Y_RESOLUTION', 'f'),
    ('Z_RESOLUTION', 'f'),
    ('NUM_R_ELEMENTS','f'),
    ('NUM_ANGLES', 'f'),
    ('Z_ROTATION_ANGLE', 'f'),
    ('DECAY_CORR_FCTR', 'f'),
    ('CORRECTIONS_APPLIED', 'l'),
    ('GATE_DURATION',  'l'),
    ('R_WAVE_OFFSET',  'l'),
    ('NUM_ACCEPTED_BEATS', 'l'),
    ('FILTER_CUTOFF_FREQUENCY','f'), 
    ('FILTER_RESOLUTION','f'),
    ('FILTER_RAMP_SLOPE','f'),
    ('FILTER_ORDER','H'),
    ('FILTER_SCATTER_CORRECTION','f'),
    ('FILTER_SCATTER_SLOPE','f'), 
    ('ANNOTATION','40s'),
    ('MT_1_1','f'), 
    ('MT_1_2','f'), 
    ('MT_1_3','f'), 
    ('MT_2_1','f'), 
    ('MT_2_2','f'), 
    ('MT_2_3','f'), 
    ('MT_3_1','f'), 
    ('MT_3_2','f'), 
    ('MT_3_3','f'), 
    ('RFILTER_CUTOFF','f'),  
    ('RFILTER_RESOLUTION','f'), 
    ('RFILTER_CODE', 'H'), 
    ('RFILTER_ORDER', 'H'),
    ('ZFILTER_CUTOFF','f'), 
    ('ZFILTER_RESOLUTION','f'), 
    ('ZFILTER_CODE', 'H'), 
    ('ZFILTER_ORDER', 'H'), 
    ('MT_4_1','f'), 
    ('MT_4_2', 'f'),
    ('MT_4_3', 'f'),
    ('SCATTER_TYPE', 'H'),
    ('RECON_TYPE', 'H'),
    ('RECON_VIEWS', 'H'),
    ('FILL', 'H')
 ))
field_formats_sh = struct_formats_sh.values();


class Ecat7(bin.BinaryFormat):
    """
    A Class to read (maybe write) ECAT7 format images.
    Generally these are PET images
    """

    # Anything which should be default different than field-defaults
    _field_defaults = {'magic_number': 'MATRIX72'}
    _sub_field_defaults = {'SCALE_FACTOR': 1.}
    
    extensions = ('.v')
    

    def __init__(self, filename, mode="r", datasource=DataSource(), **keywords):
        """
        Constructs a Ecat7 binary format object with at least a filename
        NOTE: ECAT can be an Image or a Volume
        possible additional keyword arguments:
        mode = mode to open the memmap (default is "r")
        datasource = ???
        grid = Grid object
        sctype = numpy scalar type
        intent = meaning of data
        clobber = allowed to clobber?
        
        """
        #Check if data is zipped
        if dataio.iszip(filename):
            self.filebase = dataio.unzip(filename)
        
            
        self.filebase = os.path.splitext(filename)[0]
        self.header_file = self.filebase+".v"
        self.data_file = self.filebase+".v"
        self.nframes = 0
        self.checkversion(datasource)
        
        bin.BinaryFormat.__init__(self, filename, mode, datasource, **keywords)
        self.clobber = keywords.get('clobber', False)
        self.intent = keywords.get('intent', '')

        self.header_formats = struct_formats_mh
        

        # Writing Ecat not supported yet
        if self.mode[0] is 'w':
            #Deal with writing to file or raise implement error?
            raise NotImplementedError


        # Fill Main header dict with default values
        self.byteorder = self.guess_byteorder(self.header_file,
                                                  datasource=self.datasource)
        self.header_defaults()
        self.read_header()
        self.generate_mlist(datasource)
        self.scale = self.header['ecat_calibration_factor']
        #Mlist generates nuber of frames, read in each subheader and frame data block
        self.nframes = self.mlist.shape[1]

        #self.sub_header = self.nframes * [struct_formats_sh]

        # for each frame read in subheader and attach data
        self.frames = self.nframes * [Frame]
        for i in range(self.nframes):
            self.frames[i] = Frame(self.data_file,self.byteorder,\
                                   self.mlist,self.scale,i,self.datasource)
        
        # set up canonical
        self.inform_canonical()
 
        

    def checkversion(self,datasource):
        """
        Currently only Ecat72 is implemented
        """
        hdrfile = datasource.open(self.header_file)
        hdrfile.seek(0)
        magicnumber = hdrfile.read(8)
        if magicnumber.find('MATRIX72') is -1:
            raise NotImplementedError("%s ECAT version not supported"%magic_number[0])
        

    def prewrite(self, x):
        """
        Might transform the data before writing;
        at least confirm sctype
        """
        return x.astype(self.sctype)


    def postread(self, x):
        """
        Might transform the data after getting it from memmap
        """
        return x



    @staticmethod   
    def guess_byteorder(hdrfile, datasource=DataSource()):
        """
        Determine byte order of the header.  The first header element is the
        header size.  It should always be 384.  If it is not then you know you
        read it in the wrong byte order.
        """
        if type(hdrfile)==type(""):
            hdrfile = datasource.open(hdrfile)
            hdrfile.seek(46)
            byteorder = utils.BIG_ENDIAN #Most scans are on suns = BE
            reported_length = utils.struct_unpack(hdrfile,
                                                byteorder, field_formats_mh[2])[0]
            if reported_length != SWVERSION:
                byteorder = utils.LITTLE_ENDIAN
        return byteorder

    def header_defaults(self):
        """
        Fills main header with empty default values
        """
        for field,format in self.header_formats.items():
            self.header[field] = self._default_field_value(field,format)

    @staticmethod
    def _default_field_value(fieldname, fieldformat):
        "[STATIC] Get empty defualt value for given field"
        return Ecat7._field_defaults.get(fieldname, None) or \
               utils.format_defaults[fieldformat[-1]]

    def generate_mlist(self,datasource):
        """
        List the available matricies in the ECAT file
        """
        # file.seek beyond main header, and read 512 to generate mlist
        infile = datasource.open(self.header_file)
        infile.seek(0)
        infile.seek(HEADER_SIZE)
        elements = ['128i'] # all elements are the same
        values = utils.struct_unpack(infile, self.byteorder, elements)
        values= N.reshape(values,[32,4])
        #Calculate mlist which is a matrix list with
        #  id
        #  startblock
        #  endblock
        #  status
        mlist = []
        while values[0,1] != 2:
            if values[0,0]+values[0,3] == 31:
                tmp =  values[:,1:31]
                mlist = N.asarray(mlist,tmp)
            else:
                print 'empty Mlist'
                mlist = []
                
        if mlist == []:
            mlist = values[1:values[0,3]+1]
        else:
            mlist = mlist + values[1:values[0,3]+1]
            
        self.mlist = mlist.conj().transpose()

    def inform_canonical(self, fieldsDict=None):
        tmpdat = self.frames[0]
        if fieldsDict is not None:
            self.canonical_fields = odict(fieldsDict)
        else:
            if tmpdat.subheader['DATA_TYPE'] == 1:
                self.canonical_fields['datasize'] = 8
            elif tmpdat.subheader['DATA_TYPE'] == 2 or tmpdat.subheader['DATA_TYPE'] == 6:
                self.canonical_fields['datasize'] = 16
            else:
                self.canonical_fields['datasize'] = 32

            if self.nframes > 1:
                self.canonical_fields['ndim'] = 4
                self.canonical_fields['tdim'] = self.nframes
            else:
                self.canonical_fields['ndim'] = 3
                self.canonical_fields['tdim'] = 1
                
            self.canonical_fields['xdim'] = tmpdat.subheader['X_DIMENSION']
            self.canonical_fields['ydim'] = tmpdat.subheader['Y_DIMENSION']
            self.canonical_fields['zdim'] = tmpdat.subheader['Z_DIMENSION']
            
            self.canonical_fields['scaling'] = 1


class Frame(bin.BinaryFormat):
    """
    A class to hold ECAT subheaders and associated memmaps
    """
    _sub_field_defaults = {'SCALE_FACTOR': 1.}
    
    
    def __init__(self, infile, byteorder, mlist,scale, framenumber=0, datasource=DataSource(), mode='rb'):
        bin.BinaryFormat.__init__(self, infile, mode, datasource)
        self.infile = infile
        self.data_file = infile
        self.byteorder = byteorder
        self.scale = scale
        
        self.frame = framenumber
        # sub header info
        self.subheader = odict()
        self.sub_header_formats = struct_formats_sh
        self.subheader_defaults()
        recordstart = (mlist[1][framenumber]-1)*BLOCKSIZE
        self.read_subheader(recordstart,self.datasource)

        
        self.sctype = datatype2sctype[self.subheader['DATA_TYPE']]
        self.ndim = 3
        
        ## grid for data
        if not self.grid:                
            axisnames = space[::-1]
            
            offset = (self.subheader['Z_OFFSET'],
                      self.subheader['X_OFFSET'],
                      self.subheader['Y_OFFSET'])
            step = (self.subheader['Z_PIXEL_SIZE']*10,
                      self.subheader['X_PIXEL_SIZE']*10,
                      self.subheader['Y_PIXEL_SIZE']*10)
            shape = (self.subheader['Z_DIMENSION'],
                     self.subheader['X_DIMENSION'],
                     self.subheader['Y_DIMENSION'])
            origin = [float] *3
            for i in range(len(shape)):
                origin[i] = shape[i]/2 + offset[i]
            
            
            ## Setup affine transformation        
            self.grid = SamplingGrid.from_start_step(names=axisnames,
                                                shape=shape,
                                                start=-N.array(origin)*step,
                                                step=step)
            # Get memmaped array
        offset = (mlist[1,framenumber])*BLOCKSIZE
        self.attach_data(offset)

    def subheader_defaults(self):
        """
        Fills sub header with empty default values
        """
        for field,format in self.sub_header_formats.items():
            self.subheader[field] = self._default_sub_field_value(field,format)

    
    @staticmethod
    def _default_sub_field_value(fieldname, fieldformat):
        "[STATIC] Get empty defualt value for given field"
        return Ecat7._sub_field_defaults.get(fieldname, None) or \
               utils.format_defaults[fieldformat[-1]]
        
    def read_subheader(self, recordstart,datasource=DataSource()):
        """
        Read an ECAT subheader and fill fields
        """
        infile = datasource.open(self.infile)
        infile.seek(recordstart)
        values = utils.struct_unpack(infile,
                               self.byteorder,
                               self.sub_header_formats.values())
        for field, val in zip(self.subheader.keys(), values):
            self.subheader[field] = val

    def postread(self,x):
        """
        Might transform the data after getting it from memmap
        """
        if self.subheader['SCALE_FACTOR']:
            return x * self.subheader['SCALE_FACTOR']*self.scale
        else:
            return x 
