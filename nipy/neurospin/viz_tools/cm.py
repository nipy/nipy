"""
Matplotlib colormaps useful for neuroimaging.
"""

import matplotlib as _mp

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


def _add_cmap(color_dict, cmap_dict):
    """ Construct a colormap and add it, with its inverse, to the given
        data dict.
    """


################################################################################
# Our colormaps definition

_cmaps_data = dict(
    cold_hot     = _pigtailed_cmap(_mp.cm.hot),
    brown_blue   = _pigtailed_cmap(_mp.cm.bone),
    cyan_copper  = _pigtailed_cmap(_mp.cm.copper),
    cyan_orange  = _pigtailed_cmap(_mp.cm.YlOrBr_r),
    blue_red     = _pigtailed_cmap(_mp.cm.Reds_r),
    brown_cyan   = _pigtailed_cmap(_mp.cm.Blues_r),
    purple_green = _pigtailed_cmap(_mp.cm.Greens_r,
                    swap_order=('red', 'blue', 'green')),
    purple_blue  = _pigtailed_cmap(_mp.cm.Blues_r,
                    swap_order=('red', 'blue', 'green')),
    blue_orange  = _pigtailed_cmap(_mp.cm.Oranges_r,
                    swap_order=('green', 'red', 'blue')),
    black_blue   = _rotate_cmap(_mp.cm.hot),
    black_purple = _rotate_cmap(_mp.cm.hot,
                                    swap_order=('blue', 'red', 'green')),
    black_pink   = _rotate_cmap(_mp.cm.hot,
                            swap_order=('blue', 'green', 'red')),
    black_green  = _rotate_cmap(_mp.cm.hot,
                            swap_order=('red', 'blue', 'green')),
    black_red    = _mp.cm.hot._segmentdata.copy(),
)

################################################################################
# Build colormaps and their reverse.
_cmap_d = dict()

for _cmapname in _cmaps_data.keys():
    _cmapname_r = _cmapname + '_r'
    _cmapspec = _cmaps_data[_cmapname]
    if 'red' in _cmapspec:
        _cmaps_data[_cmapname_r] = _mp.cm.revcmap(_cmapspec)
        _cmap_d[_cmapname] = _mp.colors.LinearSegmentedColormap(
                                _cmapname, _cmapspec, _mp.cm.LUTSIZE)
        _cmap_d[_cmapname_r] = _mp.colors.LinearSegmentedColormap(
                                _cmapname_r, _cmaps_data[_cmapname_r],
                                _mp.cm.LUTSIZE)
    else:
        _revspec = list(reversed(_cmapspec))
        if len(_revspec[0]) == 2:    # e.g., (1, (1.0, 0.0, 1.0))
            _revspec = [(1.0 - a, b) for a, b in _revspec]
        _cmaps_data[_cmapname_r] = _revspec

        _cmap_d[_cmapname] = _mp.colors.LinearSegmentedColormap.from_list(
                                _cmapname, _cmapspec, _mp.cm.LUTSIZE)
        _cmap_d[_cmapname_r] = _mp.colors.LinearSegmentedColormap.from_list(
                                _cmapname_r, _revspec, _mp.cm.LUTSIZE)

locals().update(_cmap_d)


