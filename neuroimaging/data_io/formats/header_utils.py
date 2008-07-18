"""Utilities convert SPM Analyze headers to Nifti1 headers.  This is a
bit of a hack, written specifically to convert a set of Nipy's test
images.  This could potentially be re-written to provide more general
functionality in the future.

Public function is get_header

"""

from neuroimaging.utils.odict import odict
from neuroimaging.data_io.formats import utils

HEADER_SIZE = 348

# Ordered dictionary of header field names mapped to struct format.
# This was pulled from nipy.data_io.formats.analyze.py Some of these
# fields do not correspond to the fields in the Analyze7.5 header and
# I'm not sure where they came from.  A few of these, like the
# scale_factor and origin are SPM extensions.  Others, like vox_units
# and cal_units, I'm not sure about there location in the header.
# However, these do work with the data set we have. ???
analyze_struct_formats = odict((
    ('sizeof_hdr','i'),
    ('data_type','10s'),
    ('db_name','18s'),
    ('extents','i'),
    ('session_error','h'),
    ('regular','c'),
    ('hkey_un0','c'),
    ('dim','8h'),
    ('vox_units','4s'),
    ('cal_units','8s'),
    ('unused1','h'),
    ('datatype','h'),
    ('bitpix','h'),
    ('dim_un0','h'),
    ('pixdim','8f'),
    ('vox_offset','f'),
    ('scale_factor','f'),
    ('funused2','f'),
    ('funused3','f'),
    ('cal_max','f'),
    ('cal_min','f'),
    ('compressed','i'),
    ('verified','i'),
    ('glmax','i'),
    ('glmin','i'),
    ('descrip','80s'),
    ('aux_file','24s'),
    ('orient','c'),
    ('origin','5h'),
    ('generated','10s'),
    ('scannum','10s'),
    ('patient_id','10s'),
    ('exp_date','10s'),
    ('exp_time','10s'),
    ('hist_un0','3s'),
    ('views','i'),
    ('vols_added','i'),
    ('start_field','i'),
    ('field_skip','i'),
    ('omax','i'),
    ('omin','i'),
    ('smax','i'),
    ('smin','i')))

analyze_field_defaults = {
    'sizeof_hdr': HEADER_SIZE,
    'extents': 16384,
    'regular': 'r',
    'hkey_un0': ' ',
    'vox_units': 'mm',
    'scale_factor': 1.0 }

def get_byteorder(hdrfile):
    """Return the byteorder from the file."""
    byteorder = utils.LITTLE_ENDIAN
    # for analyze 7.5 and nifti1 first field is a 4-byte int, sizeof_hdr
    reported_length = utils.struct_unpack(hdrfile, byteorder, 'i')[0]
    if reported_length != HEADER_SIZE:
        byteorder = utils.BIG_ENDIAN
    return byteorder
    
def analyze_default_field_value(fieldname, fieldformat):
    """Get the default value for the given field."""
    return analyze_field_defaults.get(fieldname, None) or \
           utils.format_defaults[fieldformat[-1]]

def default_analyze_header():
    """Return an analyze header with default fields filled in."""
    fmts = analyze_struct_formats.copy()
    hdr = odict()
    for field, format in fmts.items():
        hdr[field] = analyze_default_field_value(field, format)
    return hdr

def get_header(filename, byteorder=None):
    """Get the header from `filename`.

    Parameters
    ----------
    filename : string
    byteorder : [None, '<', '>']
        If None, it will be determined from file.
        '<' is Little Endian
        '>' is Big Endian

    Returns
    -------
    hdr : dictionary

    """
    
    fp = file(filename)
    hdr = default_analyze_header()
    if byteorder is None:
        byteorder = get_byteorder(fp)
    fp.seek(0)
    fmts = analyze_struct_formats.copy()
    values = utils.struct_unpack(fp, byteorder, fmts.values())
    for field, val in zip(hdr.keys(), values):
        hdr[field] = val
    return hdr
