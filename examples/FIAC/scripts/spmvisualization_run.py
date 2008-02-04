"""
python visualization_run.py [design,contrast,contrast_type]
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import spm_model.spm_visualization as visualization

def run(design, contrast, which):
    visualization.run(contrast=contrast,
                      which=which,
                      design=design)

if __name__ == "__main__":
    import sys
    design, contrast, which = sys.argv[1:]    
    run(design, contrast, which)
