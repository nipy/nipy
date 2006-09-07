

import os,sys
import struct
import numpy

CHAR = 8
UINT16 = 16
UINT32 = 32
FLOAT32 = 32
ENDIAN = '>' # native endian

#ENDIAN

mainheader =  [('magic_number', '14s', 'MATRIX72'),
               ('original_file_name', '32s',''),
               ('sw_version', 'H', 0),
               ('system_type', 'H', 0),
               ('file_type', 'H', 0),
               ('serial_number', '10s', ''),
               ('scan_start_time', 'l',0),
               ('isotope_name', '8s', ''),
               ('isotope_halflife', 'f',0),
               ('radiopharmaceutical', '32s', ''),
               ('gantry_tilt', 'f', 0),
               ('gantry_rotation', 'f',0),
               ('bed_elevation', 'f',0),
               ('intrinsic_tilt', 'f', 0),           
               ('wobble_speed', 'H', 0),             
               ('transm_source_type', 'H',0),
               ('distance_scanned', 'f', 0),
               ('transaxial_fov', 'f',0),           
               ('angular_compression', 'H', 0),
               ('coin_samp_mode', 'H', 0),
               ('axial_samp_mode', 'H', 0),          
               ('ecat_calibration_factor','f',0),
               ('calibration_units', 'H',0),        
               ('calibration_units_type','H',0),   
               ('compression_code', 'H', 0),             
               ('study_type', '12s', ''),              
               ('patient_id', '16s', ''),               
               ('patient_name', '32s', ''),             
               ('patient_sex', 's', ''),              
               ('patient_dexterity', 's', ''),        
               ('patient_age', 'f', 0),              
               ('patient_height', 'f', 0),           
               ('patient_weight', 'f', 0),           
               ('patient_birth_date', 'I',0),       
               ('physician_name', '32s', ''),
               ('operator_name', '32s', ''),          
               ('study_description', '32s',''),        
               ('acquisition_type', 'H', 0),        
               ('patient_orientation', 'H', 0),     
               ('facility_name', '20s', ''),            
               ('num_planes', 'H', 0),               
               ('num_frames', 'H', 0),               
               ('num_gates', 'H', 0),               
               ('num_bed_pos', 'H', 0),    
               ('init_bed_position', 'f', 0),
               ('bed_position(1)', 'f', 0),
               ('bed_position(2)', 'f', 0),
               ('bed_position(3)', 'f',0),
               ('bed_position(4)', 'f',0),  
               ('bed_position(5)', 'f',0),  
               ('bed_position(6)', 'f',0),  
               ('bed_position(7)', 'f',0),  
               ('bed_position(8)', 'f',0),  
               ('bed_position(9)', 'f',0),  
               ('bed_position(10)', 'f',0),  
               ('bed_position(11)', 'f',0),  
               ('bed_position(12)', 'f',0),  
               ('bed_position(13)', 'f',0),  
               ('bed_position(14)', 'f',0),  
               ('bed_position(15)', 'f',0),  
               ('plane_separation', 'f',0),
               ('lwr_sctr_thres', 'H',0), 
               ('lwr_true_thres', 'H',0),           
               ('upr_true_thres', 'H',0),           
               ('user_process_code', '10s', ''),
               ('acquisition_mode', 'H', 0),
               ('bin_size', 'f',0),                 
               ('branching_fraction', 'f',0),       
               ('dose_start_time', 'f',0),          
               ('dosage', 'f',0),                   
               ('well_counter_corr_factor', 'f',0), 
               ('data_units', '32s', ''), 
               ('septa_state', 'H', 0),   
               ('fill', 'H',0)            
               ]

fdg_ecatfile = '/home/surge/cindeem/DEVEL/RAW_PET/FDG/B05_235-42A890F600000F79-de.v'
infile = open(fdg_ecatfile, 'rb')

byteoffset = (0,
              14,
              46,
              48,
              50,
              52,
              62,
              66,
              74,
              78,
              110,
              114,
              118,
              122,
              126,
              128,
              130,
              134,
              138,
              140,
              142,
              144,
              148,
              150,
              152,
              154,
              166,
              182,
              214,
              215,
              216,
              220,
              224,
              228,
              232,
              264,
              296,
              328,
              330,
              332,
              352,
              354,
              356,
              358,
              360,
              364,
              368,
              372,
              376,
              380,
              384,
              388,
              392,
              396,
              400,
              404,
              408,
              412,
              416,
              420,
              424,
              428,
              430,
              432,
              434,
              444,
              446,
              450,
              454,
              458,
              462,
              466,
              498,
              500,
              )
bytesize = (14,
32,
2,
2,
2,
10,
4,
8,
4,
32,
4,
4,
4,
4,
2,
2,
4,
4,
2,
2,
2,
4,
2,
2,
2,
12,
16,
32,
1,
1,
4,
4,
4,
4,
32,
32,
32,
2,
2,
20,
2,
2,
2,
2,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
2,
2,
2,
10,
2,
4,
4,
4,
4,
4,
32,
2,
2,
)

hdrindex = range(74)

for i in hdrindex:
    tmp = mainheader[i]
    vtype = tmp[1]
    infile.seek(byteoffset[i])
    tmpin = infile.read(bytesize[i])
    vtype = ENDIAN + vtype
    tmpin2 = struct.unpack(vtype,tmpin)
    if vtype.find('s') >= 0:
        tmpstr= tmpin2[0]
        tmpin2 = ( tmpstr.replace('\x00',''),)
    tmp = tmp[0:2] + (tmpin2)
    mainheader[i]= tmp


infile.seek(512) # go to subheader
tmpblock = infile.read(512)
block = struct.unpack('>128i',tmpblock)
block = numpy.reshape(block,[32,4])


list = []
while block[0,1] != 2:
    if block[0,0]+block[0,3] == 31:
        tmp =  block[:,1:31]
        list = numpy.asarray(list,tmp)
    else:
        print 'empty List'
        list = []

if list == []:
    list = block[1:block[0,3]+1]
else:
   list = list + block[1:block[0,3]+1]

list = list.conj().transpose()
# list is matrix list
# id
# startblock
# endblock
# status
                 

subheader = [('DATA_TYPE', 'H',0),
             ('NUM_DIMENSIONS','H',0),
             ('X_DIMENSION','H',0),
             ('Y_DIMENSION','H',0),
             ('Z_DIMENSION','H',0),
             ('X_OFFSET','f',0),
             ('Y_OFFSET','f',0),
             ('Z_OFFSET', 'f',0),
             ('RECON_ZOOM','f',0),
             ('SCALE_FACTOR','f',0),
             ('IMAGE_MIN', 'h',0),
             ('IMAGE_MAX', 'h', 0),
             ('X_PIXEL_SIZE', 'f',0), 
             ('Y_PIXEL_SIZE', 'f',0), 
             ('Z_PIXEL_SIZE', 'f',0), 
             ('FRAME_DURATION', 'f',0), 
             ('FRAME_START_TIME', 'f',0),
             ('FILTER_CODE', 'H',0),
             ('X_RESOLUTION','f',0),
             ('Y_RESOLUTION', 'f',0),
             ('Z_RESOLUTION', 'f',0),
             ('NUM_R_ELEMENTS','f',0),
             ('NUM_ANGLES', 'f',0),
             ('Z_ROTATION_ANGLE', 'f',0),
             ('DECAY_CORR_FCTR', 'f',0),
             ('CORRECTIONS_APPLIED', 'l',0),
             ('GATE_DURATION',  'l',0),
             ('R_WAVE_OFFSET',  'l',0),
             ('NUM_ACCEPTED_BEATS', 'l',0),
             ('FILTER_CUTOFF_FREQUENCY','f',0), 
             ('FILTER_RESOLUTION','f',0),
             ('FILTER_RAMP_SLOPE','f',0),
             ('FILTER_ORDER','H',0),
             ('FILTER_SCATTER_CORRECTION','f',0),
             ('FILTER_SCATTER_SLOPE','f',0), 
             ('ANNOTATION','40s',''),
             ('MT_1_1','f',0), 
             ('MT_1_2','f',0), 
             ('MT_1_3','f',0), 
             ('MT_2_1','f',0), 
             ('MT_2_2','f',0), 
             ('MT_2_3','f',0), 
             ('MT_3_1','f',0), 
             ('MT_3_2','f',0), 
             ('MT_3_3','f',0), 
             ('RFILTER_CUTOFF','f',0),  
             ('RFILTER_RESOLUTION','f',0), 
             ('RFILTER_CODE', 'H',0), 
             ('RFILTER_ORDER', 'H',0),
             ('ZFILTER_CUTOFF','f',0), 
             ('ZFILTER_RESOLUTION','f',0), 
             ('ZFILTER_CODE', 'H',0), 
             ('ZFILTER_ORDER', 'H',0), 
             ('MT_4_1','f',0), 
             ('MT_4_2', 'f',0),
             ('MT_4_3', 'f',0),
             ('SCATTER_TYPE', 'H',0),
             ('RECON_TYPE', 'H',0),
             ('RECON_VIEWS', 'H',0),
             ('FILL', 'H',0)]

subbyteoffset = (0,
2,
4,
6,
8,
10,
14,
18,
22,
26,
30,
32,
34,
38,
42,
46,
50,
54,
56,
60,
64,
68,
72,
76,
80,
84,
88,
92,
96,
100,
104,
108,
112,
114,
118,
122,
162,
166,
170,
174,
178,
182,
186,
190,
194,
198,
202,
206,
208,
210,
214,
218,
220,
222,
226,
230,
234,
236,
238,
240)

subbytesize= (2,
2,
2,
2,
2,
4,
4,
4,
4,
4,
2,
2,
4,
4,
4,
4,
4,
2,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
2,
4,
4,
40,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
4,
2,
2,
4,
4,
2,
2,
4,
4,
4,
2,
2,
2,
2)

sheaders = []
for index1 in range(list.shape[1]):
    recordstart = (list[1,index1]-1)* 512
    for hdrindex in range(len(subheader)):
        tmp = subheader[hdrindex]
        vtype = tmp[1]
        infile.seek(recordstart + subbyteoffset[hdrindex])
        tmpin = infile.read(subbytesize[hdrindex])
        vtype = ENDIAN + vtype
        tmpin2 = struct.unpack(vtype,tmpin)
        if vtype.find('s') >= 0:
            tmpstr= tmpin2[0]
            tmpin2 = ( tmpstr.replace('\x00',''),)

        tmp = tmp[0:2] + (tmpin2)
        subheader[hdrindex]= tmp

    sheaders = sheaders + subheader
            
## Read in DATA blocks
BLKSIZE = 512
FIRSTBLK = 2

infile.seek((FIRSTBLK-1)* BLKSIZE)
sn = infile.read(BLKSIZE/4)
numbytes = str(128/4)
readtype = ENDIAN + numbytes+ 'i'
dirbuf = struct.unpack(readtype,sn)

for blockindex in range(list.shape[1]):
    blockNR = list[2,blockindex] - list[1,blockindex]
    infile.seek((list[1,blockindex]-1)*BLKSIZE)
    rawDataHex = infile.read(blockNR*512)
    matrixDataType = ENDIAN + str(blockNR*512/2) + 'H'
    
    rawData = struct.unpack(matrixDataType,rawDataHex)

infile.close()

