import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))


import model, io, fixed # FIAC.model, FIAC.io, FIAC.fixed

from spm_model import Run

class Fixed(fixed.Fixed):

    def resultpath(self, path):
        return os.path.join(self.root, 'fixed-spm', self.design, self.which, self.contrast, path)

class Subject(fixed.Subject):

    def __init__(self, id, contrast):
        self.id = id
        model.Subject.__init__(self, id, study=contrast)
        runs = []
 
        for run in getattr(self, self.study.design):
            runmodel = Run(self, run)
            runs.append(runmodel)
        self.runs = runs

        if not os.path.exists(self.resultpath("")):
            os.makedirs(self.resultpath(""))

def run(root=io.data_path, subj=3, resample=True, fit=True):

    for contrast in ['average', 'interaction', 'speaker', 'sentence']:
        for which in ['contrasts', 'delays']:
            for design in ['event', 'block']:
                fixed_ = Fixed(root=io.data_path, which=which, contrast=contrast, design=design)
                subject = Subject(subj, fixed_)

                if fit:
                    effect, sd, t = subject.fit()
