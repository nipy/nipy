from enthought.traits.ui import View, Group, Item, Handler, message
from enthought.traits.ui.menu import MenuBar, Menu, Action, Separator, OKCancelButtons
from enthought.pyface.image_resource import ImageResource
from neuroimaging import traits

import sys, os, urllib2
sys.path.append(os.path.abspath('..'))

import run as FIACrun
import fiac, pylab
import numpy as N
import time

base = 'http://kff.stanford.edu/FIAC/'

class RunUI(traits.HasTraits):

    # Subject/run traits

    run = traits.Range(low=1, high=5, value=1)
    subj = traits.Range(low=0, high=15, value=0)
    mask = traits.true

    fmrifile = traits.Str
    maskfile = traits.Str
    
    # Confound traits

    normalize = traits.true
    mean_reg = traits.true
    drift_df = traits.Int
    knots = traits.Array(shape=(None,))
    tmax = traits.Float(500)
    tmin = traits.Float(0)
    blah = traits.Event

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self._subj_changed()
        self._drift_df_changed()
        
    def _drift_df_changed(self):
        self.knots = N.linspace(self.tmin, self.tmax, self.drift_df - 2)[1:-1]

    def _subj_changed(self):
        self.maskfile = fiac.FIACpath('fsl/mask.img', subj=self.subj, run=self.run, base=base)
        if not self.validate():
            self.fmrifile = 'URL "%s" not found' % self.fmrifile
            self.maskfile = 'URL "%s" not found' % self.maskfile

    def validate(self):
        self.fmrifile = fiac.FIACpath('fsl/filtered_func_data.img', subj=self.subj, run=self.run, base=base)
        try:
            test = urllib2.urlopen(self.fmrifile)
            return True
        except:
            return False

    def _run_changed(self):
        self._subj_changed()

       
class MyHandler(Handler):

    data_group = Group(
            Item(
                object = 'RunUI',
                name   = 'subj',
                label  = 'Subject'
            ),
            Item(
                object = 'RunUI',
                name   = 'run',
                label  = 'run'
            ),
            Item(
                object = 'RunUI',
                name   = 'mask',
                label  = 'Use mask?'
            ),
            Item(
                name   = '_',
            ),
            Item(
                object = 'RunUI',
                name   = 'fmrifile',
                label  = 'Data:',
                style  = 'readonly'
            ),
            Item(
                object = 'RunUI',
                name   = 'maskfile',
                label  = 'Mask:',
                style  = 'readonly'
            ),
            label = 'Data')

    confound_group = Group(
            Item(
                object = 'RunUI',
                name   = 'drift_df',
                label  = 'Spline DF'
            ),
            Item(
                object = 'RunUI',
                name   = 'knots',
                label  = 'Spline knots',
                style  = 'readonly'
            ),
            Item( name = '_' ),
            Item(
                object = 'RunUI',
                name   = 'normalize',
                tooltip = 'Normalize frames to % bold?'
            ),
            Item(
                object = 'RunUI',
                name   = 'mean_reg',
                label  = 'Mean regressor',
                tooltip = 'Use frame averages as a regressor?'
            ),
            label = 'Confounds')

    menubar = MenuBar(
            Menu(
                Action(
                    id     = 'designplot',
                    name   = 'View design',
                    action = 'do_designplot'
                ),
                Action(
                    id     = 'fitmodel',
                    name   = 'Fit model',
                    action = 'do_fitmodel'
                ),
                name = 'Actions'
            )
        )

    def do_designplot(self, info):

        run_ui = info.ui.context['RunUI']
        if run_ui.validate():
            message("Matplotlib doesn't work with envisage, does it?",
                    buttons=['OK'], title='Would be nice')
        else:
            message("Even if matplotlib worked, we have no data...",
                    buttons=['OK'], title='Nice try')

    def do_fitmodel(self, info):

        run_ui = info.ui.context['RunUI']
        if run_ui.validate():
            message("Click to begin analysis.",
                    buttons=['OK'], title="Let's go")

            toc = time.time()
            FIACrun.FIACrun(subj=run_ui.subj, run=run_ui.run)            
            tic = time.time()

            message("Analysis complete. Time elapsed %d seconds" % tic-toc,
                    buttons=['OK'], title="Finished")
            
        else:
            message("Still no data...",
                    buttons=['OK'], title='Nice try')

    traits_view = View(data_group, confound_group, menubar=menubar,
                       buttons=OKCancelButtons, title='FIAC single run')

    data_view = View(data_group)
    confound_view = View(confound_group)


class Message ( traits.HasTraits ):
    
    message = traits.Str 

"""
This example is from ui.message.
"""

def message ( message = '', title = 'Message', buttons = [ 'OK' ],
              parent  = None ):
    msg = Message( message = message )
    ui  = msg.edit_traits(parent = parent,
                          view   = View(Item(name   = 'message',
                                              style  = 'readonly',
                                              label  = ''),
                                        title   = title,
                                        buttons = buttons,
                                        kind    = 'modal'))

    return ui.result


if __name__ == '__main__':
    a = RunUI()
    handler = MyHandler()

    a.handler = handler
    handler.configure_traits(context={'RunUI':a})

    print 'run', a.run

    handler.configure_traits(view='data_view', context={'RunUI':a})
    print 'run', a.run

