import numpy, types, time

AFNI_missing = [-999,-999999]

AFNI_float = 'float-attribute'
AFNI_integer = 'integer-attribute'
AFNI_string = 'string-attribute'

AFNI_UChar = 0
AFNI_Short = 1
AFNI_Float = 3
AFNI_Complex = 5

AFNI_brick_types = {AFNI_UChar:numpy.uint8,
                   AFNI_Short:numpy.int16,
                   AFNI_Float:numpy.float32,
                   AFNI_Complex:numpy.complex64
                   }

AFNI_view = {'orig':0,
             'acpc':1,
             'tlrc':2
             }

AFNI_typestring = {'3DIM_HEAD_ANAT':0,
                   '3DIM_HEAD_FUNC':1,
                   '3DIM_GEN_ANAT':2,
                   '3DIM_GEN_FUNC':3
                   }
                   
ANAT_SPGR_TYPE = 0
ANAT_FSE_TYPE = 1
ANAT_EPI_TYPE = 2
ANAT_MRAN_TYPE = 3
ANAT_CT_TYPE = 4
ANAT_SPECT_TYPE = 5
ANAT_PET_TYPE = 6
ANAT_MRA_TYPE = 7
ANAT_BMAP_TYPE = 8
ANAT_DIFF_TYPE = 9
ANAT_OMRI_TYPE = 10
ANAT_BUCK_TYPE = 11

FUNC_FIM_TYPE = 0   #   1 value           
FUNC_THR_TYPE = 1   #  obsolete          
FUNC_COR_TYPE = 2   #  fico: correlation 
FUNC_TT_TYPE = 3    #  fitt: t-statistic 
FUNC_FT_TYPE = 4    #  fift: F-statistic 
FUNC_ZT_TYPE = 5    #  fizt: z-score     
FUNC_CT_TYPE = 6    #  fict: Chi squared 
FUNC_BT_TYPE = 7    #  fibt: Beta stat   
FUNC_BN_TYPE = 8    #  fibn: Binomial    
FUNC_GT_TYPE = 9    #  figt: Gamma       
FUNC_PT_TYPE = 10   #  fipt: Poisson     
FUNC_BUCK_TYPE = 11 #  fbuc: bucket      

ORI_R2L_TYPE = 0  # Right to Left         
ORI_L2R_TYPE = 1  # Left to Right         
ORI_P2A_TYPE = 2  # Posterior to Anterior 
ORI_A2P_TYPE = 3  # Anterior to Posterior 
ORI_I2S_TYPE = 4  # Inferior to Superior  
ORI_S2I_TYPE = 5  # Superior to Inferior  

UNITS_MSEC_TYPE = 77001
UNITS_SEC_TYPE = 77002
UNITS_HZ_TYPE = 77003  

AFNI_byteorder = {'big':'MSB_FIRST',
                  'little':'LSB_FIRST'
                  }

# Below ignores the "variable" named attributed NOTE_NUMBER_xxx, etc.

AFNI_header_atts = {'DATASET_RANK':(types.IntType, None),
                    'DATASET_DIMENSIONS':(types.IntType, None),
                    'TYPESTRING':(types.StringType, None),
                    'SCENE_DATA':(types.IntType, None),
                    'ORIENT_SPECIFIC':(types.IntType, 3),
                    'ORIGIN':(types.FloatType, 3),
                    'DELTA':(types.FloatType, 3),
                    'TAXIS_NUMS':(types.IntType, 3),
                    'TAXIS_FLOATS':(types.FloatType, 5),
                    'TAXIS_OFFSETS':(types.FloatType, None), 
                    'IDCODE_STRING':(types.StringType, None),
                    'IDCODE_DATE':(types.StringType, None),
                    'BYTEORDER_STRING':(types.StringType, None),
                    'BRICK_STATS':(types.FloatType, None),
                    'BRICK_TYPES':(types.IntType, None),
                    'BRICK_FLOAT_FACS':(types.FloatType, None),
                    'BRICK_LABS':(types.StringType, None),
                    'BRICK_STATAUX':(types.FloatType, None),
                    'STAT_AUX':(types.FloatType, None),
                    'HISTORY_NOTE':(types.StringType, None),
                    'NOTES_COUNT':(types.IntType, 1),
                    'TAGALIGN_MATVEC':(types.FloatType, 12),
                    'VOLREG_ROTCOM':(types.StringType, None),
                    'VOLREG_CENTER_OLD':(types.StringType, None),
                    'VOLREG_CENTER_BASE':(types.StringType, None),
                    'VOLREG_ROTPARENT_IDCODE':(types.StringType, None),
                    'VOLREG_ROTPARENT_NAME':(types.StringType, None),
                    'VOLREG_GRIDPARENT_IDCODE':(types.StringType, None),
                    'VOLREG_GRIDPARENT_NAME':(types.StringType, None),
                    'VOLREG_INPUT_IDCODE':(types.StringType, None),
                    'VOLREG_INPUT_NAME':(types.StringType, None),
                    'VOLREG_BASE_IDCODE':(types.StringType, None),
                    'VOLREG_BASE_NAME':(types.StringType, None),
                    'VOLREG_ROTCOM_NUM':(types.IntType, 1),
                    'IDCODE_ANAT_PARENT':(types.StringType, None),
                    'TO3D_ZPAD':( types.IntType, 3),
                    'IDCODE_WARP_PARENT':(types.StringType, None),
                    'WARP_TYPE':(types.IntType, 2),
                    'WARP_DATA':(types.FloatType, None),
                    'MARKS_XYZ':(types.FloatType, 30),
                    'MARKS_LAB':(types.StringType, None),
                    'MARKS_FLAGS':(types.IntType, None),
                    'TAGSET_NUM':(types.IntType, 2),
                    'TAGSET_FLOATS':(types.FloatType, None),
                    'TAGSET_LABELS':(types.StringType, None),
                    'LABEL_1':(types.StringType, None),
                    'LABEL_2':(types.StringType, None),
                    'DATASET_NAME':(types.StringType, None),
                    'DATASET_KEYWORD':(types.StringType, None),
                    'BRICK_KEYWORDS':(types.StringType, None)
                    }

AFNI_orientations = {('xspace',True): ORI_R2L_TYPE,
                     ('xspace',False): ORI_L2R_TYPE,
                     ('yspace',True): ORI_A2P_TYPE,
                     ('yspace',False): ORI_P2A_TYPE,
                     ('zspace',True): ORI_I2S_TYPE,
                     ('zspace',False): ORI_S2I_TYPE
                     }

AFNI_orientations_inv = {ORI_L2R_TYPE: ('xspace', False),
                         ORI_R2L_TYPE: ('xspace', True),
                         ORI_P2A_TYPE: ('yspace', True),
                         ORI_A2P_TYPE: ('yspace', False),
                         ORI_I2S_TYPE: ('zspace', True),
                         ORI_S2I_TYPE: ('zspace', False)}

