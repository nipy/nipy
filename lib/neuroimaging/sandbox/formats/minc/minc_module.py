import numpy as N
from scipy.weave import ext_tools


def build_minc_module():
    """
    Builds an extension module for MINC.
    """

    extension_code = """

    char *_strcp(std::string in) {
    size_t n; 
    char *out;
    n = strlen(in.c_str());
    out = (char *) malloc(sizeof(*out) * n);
    strcpy(out, in.c_str());
    return(out);
    }

    """

    mod = ext_tools.ext_module('_minc', compiler='gcc')
    mod.customize.add_header('<minc.h>')
    mod.customize.add_header('<string.h>')
    mod.customize.add_support_code(extension_code)

    # Open a MINC file

    filename = "out.mnc"
    mode = 0
    open_code = """
    char *mincfile;

    mincfile = _strcp(filename);
    return_val = miopen(mincfile, mode);
    free(mincfile);

    """

    miopen = ext_tools.ext_function('_miopen',
                                    open_code,
                                    ['filename', 'mode'])
    mod.add_function(miopen)

    # Close a MINC file

    mincid = 3
    close_code = """
    return_val = miclose(mincid);
    """

    miclose = ext_tools.ext_function('_miclose',
                                     close_code,
                                     ['mincid'])

    mod.add_function(miclose)

    # Add a standard MINC1 dimension

    filename = 'out.mnc'
    dimname = 'xspace'
    dimlen = 40
    step = 1.0
    start = 0.0
    dcos = N.array([1,0,0], N.float64)

    add_dim_code = """
    int dimid, varid, cdfid;
    char *mincfile, *_dimname;

    mincfile = _strcp(filename);
    _dimname = _strcp(dimname);

    cdfid = miopen(mincfile, NC_WRITE);

    (void) ncredef(cdfid);
    dimid  = ncdimdef(cdfid, _dimname, dimlen);

    /* Create the variable if needed */
    varid = micreate_std_variable(cdfid, _dimname,
                                  NC_INT, 0, NULL);
    (void) miattputdbl(cdfid, varid, MIstep, step);
    (void) miattputdbl(cdfid, varid, MIstart, start);
	 
    (void) ncattput(cdfid, varid, MIdirection_cosines, NC_DOUBLE,
	            Ndcos[0], dcos);
    free(mincfile);
    free(_dimname);
    
    ncendef(cdfid);
    miclose(cdfid);

    """
    
    add_dim = ext_tools.ext_function('_add_dim',
                                     add_dim_code,
                                     ['filename', 'dimname', 'dimlen', 'start', 'step', 'dcos'])

    mod.add_function(add_dim)

    # Add time dimension to MINC file


    filename = 'out.mnc'
    dimname = 'xspace'
    dimlen = 40
    step = 1.0
    start = 0.0
    dcos = N.array([1,0,0], N.float64)

    add_time_code = """
    int dimid, varid, cdfid;
    char *mincfile, *_dimname;

    mincfile = _strcp(filename);
    _dimname = _strcp(dimname);

    cdfid = miopen(mincfile, NC_WRITE);

    (void) ncredef(cdfid);
    dimid  = ncdimdef(cdfid, _dimname, dimlen);

    /* Create the variable if needed */
    varid = micreate_std_variable(cdfid, _dimname,
                                  NC_INT, 0, NULL);
    (void) miattputdbl(cdfid, varid, MIstep, step);
    (void) miattputdbl(cdfid, varid, MIstart, start);
	 
    (void) ncattput(cdfid, varid, MIdirection_cosines, NC_DOUBLE,
	            Ndcos[0], dcos);
    free(mincfile);
    free(_dimname);

    ncendef(cdfid);
    miclose(cdfid);

    """


    return mod

try:
    import _minc
except ImportError:
    mod = build_minc_module()
    mod.compile(libraries=['minc', 'netcdf'])
    import _minc

i = _minc._miopen('/home/jtaylo/Desktop/tmc_T2.mnc', 1)
_minc._miclose(i)

print i
