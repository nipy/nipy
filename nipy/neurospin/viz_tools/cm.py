# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matplotlib colormaps useful for neuroimaging.
"""

from matplotlib import cm as _cm
from matplotlib import colors as _colors

################################################################################
# Custom colormaps for two-tailed symmetric statistics
################################################################################

################################################################################
# Helper functions 

def _rotate_cmap(cmap, swap_order=('green', 'red', 'blue')):
    """ Utility function to swap the colors of a colormap.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[0]]]
    cdict['blue'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[1]]]
    cdict['red'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[2]]]

    return cdict


def _pigtailed_cmap(cmap, swap_order=('green', 'red', 'blue')):
    """ Utility function to make a new colormap by concatenating a
        colormap with its reverse.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[0]])]
    cdict['blue'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[1]])]
    cdict['red'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[2]])]

    for color in ('red', 'green', 'blue'):
        cdict[color].extend([(0.5*(1+p), c1, c2) 
                                    for (p, c1, c2) in orig_cdict[color]])

    return cdict


################################################################################
# Our colormaps definition

_cmaps_data = dict(
    cold_hot     = _pigtailed_cmap(_cm.hot),
    brown_blue   = _pigtailed_cmap(_cm.bone),
    cyan_copper  = _pigtailed_cmap(_cm.copper),
    cyan_orange  = _pigtailed_cmap(_cm.YlOrBr_r),
    blue_red     = _pigtailed_cmap(_cm.Reds_r),
    brown_cyan   = _pigtailed_cmap(_cm.Blues_r),
    purple_green = _pigtailed_cmap(_cm.Greens_r,
                    swap_order=('red', 'blue', 'green')),
    purple_blue  = _pigtailed_cmap(_cm.Blues_r,
                    swap_order=('red', 'blue', 'green')),
    blue_orange  = _pigtailed_cmap(_cm.Oranges_r,
                    swap_order=('green', 'red', 'blue')),
    black_blue   = _rotate_cmap(_cm.hot),
    black_purple = _rotate_cmap(_cm.hot,
                                    swap_order=('blue', 'red', 'green')),
    black_pink   = _rotate_cmap(_cm.hot,
                            swap_order=('blue', 'green', 'red')),
    black_green  = _rotate_cmap(_cm.hot,
                            swap_order=('red', 'blue', 'green')),
    black_red    = _cm.hot._segmentdata.copy(),
)

################################################################################
# Build colormaps and their reverse.
_cmap_d = dict()

for _cmapname in _cmaps_data.keys():
    _cmapname_r = _cmapname + '_r'
    _cmapspec = _cmaps_data[_cmapname]
    if 'red' in _cmapspec:
        _cmaps_data[_cmapname_r] = _cm.revcmap(_cmapspec)
        _cmap_d[_cmapname] = _colors.LinearSegmentedColormap(
                                _cmapname, _cmapspec, _cm.LUTSIZE)
        _cmap_d[_cmapname_r] = _colors.LinearSegmentedColormap(
                                _cmapname_r, _cmaps_data[_cmapname_r],
                                _cm.LUTSIZE)
    else:
        _revspec = list(reversed(_cmapspec))
        if len(_revspec[0]) == 2:    # e.g., (1, (1.0, 0.0, 1.0))
            _revspec = [(1.0 - a, b) for a, b in _revspec]
        _cmaps_data[_cmapname_r] = _revspec

        _cmap_d[_cmapname] = _colors.LinearSegmentedColormap.from_list(
                                _cmapname, _cmapspec, _cm.LUTSIZE)
        _cmap_d[_cmapname_r] = _colors.LinearSegmentedColormap.from_list(
                                _cmapname_r, _revspec, _cm.LUTSIZE)

locals().update(_cmap_d)


################################################################################
# Utility to replace a colormap by another in an interval
################################################################################

def dim_cmap(cmap, factor=.3):
    """ Dim a colormap to white.
    """
    assert factor >= 0 and factor <=1, ValueError(
            'Dimming factor must be larger than 0 and smaller than 1, %s was passed.' 
                                                        % factor)
    
    cdict = cmap._segmentdata.copy()
    for c_index, color in enumerate(('red', 'green', 'blue')):
        color_lst = list()
        for value, c1, c2 in cdict[color]:
            color_lst.append((value, 1 - factor*(1-c1), 
                                     1 - factor*(1-c2)))
        cdict[color] = color_lst

    return _colors.LinearSegmentedColormap(
                                '%s_dimmed' % cmap.name,
                                cdict,
                                _cm.LUTSIZE)


def replace_inside(outer_cmap, inner_cmap, vmin, vmax):
    """ Replace a colormap by another inside a pair of values.
    """
    assert vmin < vmax, ValueError('vmin must be smaller than vmax')
    assert vmin >= 0,    ValueError('vmin must be larger than 0, %s was passed.' 
                                        % vmin)
    assert vmax <= 1,    ValueError('vmax must be smaller than 1, %s was passed.' 
                                        % vmax)
    outer_cdict = outer_cmap._segmentdata.copy()
    inner_cdict = inner_cmap._segmentdata.copy()

    cdict = dict()
    for c_index, color in enumerate(('red', 'green', 'blue')):
        color_lst = list()
        for value, c1, c2 in outer_cdict[color]:
            if value >= vmin:
                break
            color_lst.append((value, c1, c2))

        color_lst.append((vmin, outer_cmap(vmin)[c_index], 
                                inner_cmap(vmin)[c_index]))

        for value, c1, c2 in inner_cdict[color]:
            if value <= vmin:
                continue
            if value >= vmax:
                break
            color_lst.append((value, c1, c2))

        color_lst.append((vmax, inner_cmap(vmax)[c_index],
                                outer_cmap(vmax)[c_index]))

        for value, c1, c2 in outer_cdict[color]:
            if value <= vmax:
                continue
            color_lst.append((value, c1, c2))

        cdict[color] = color_lst

    return _colors.LinearSegmentedColormap(
                                '%s_inside_%s' % (inner_cmap.name, outer_cmap.name),
                                cdict,
                                _cm.LUTSIZE)


