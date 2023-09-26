# Do not collect tests for files depending on Matplotlib.
try:
    import matplotlib
except ImportError:
    collect_ignore = [
        'viz.py',
        'viz_tools/activation_maps.py',
        'viz_tools/cm.py',
        'viz_tools/slicers.py',
    ]
