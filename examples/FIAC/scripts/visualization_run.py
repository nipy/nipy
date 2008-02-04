"""
python visualization_run.py [design,contrast,contrast_type]
"""

import visualization, compare

def run(design, contrast, which):

    visualization.run(contrast=contrast,
                      which=which,
                      design=design)

    compare.visualization_run(contrast=contrast,
                              which=which,
                              design=design)

if __name__ == "__main__":
    import sys
    design, contrast, which = sys.argv[1:]    
    run(design, contrast, which)
