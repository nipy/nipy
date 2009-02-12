.. _repository_api:

Repository API
==============

See also :ref:`repository_design` and :ref:`brainvisa_repositories`

FMRI datasets often have the structure:

* Group (sometimes) e.g. Patients, Controls

  * Subject e.g. Subject1, Subject2

    * Session e.g. Sess1, Sess1

How about an interface like:

::

   repo = GSSRespository(
       root_dir = '/home/me/data/experiment1',
       groups = {'patients':
                 {'subjects':
                  {'patient1':
                   {'sess1':
                    'filter': 'raw*nii'},
                   {'sess2':
                    'filter': 'raw*nii'}
                   },
                  {'patient2':
                   {'sess1':
                    'filter': 'raw*nii'}
                   {'sess2':
                    'filter': 'raw*nii'}
                   }
                  },
                 'controls':
                 {'subjects':
                  {'control1':
                   {'sess1':
                    'filter': 'raw*nii'},
                   {'sess2':
                    'filter': 'raw*nii'}
                   },
                  {'control2':
                   {'sess1':
                    'filter': 'raw*nii'}
                   {'sess2':
                    'filter': 'raw*nii'}
                   }
                  }
                 })
   
   for group in repo.groups:
       for subject in group.subjects:
           for session in subject.sessions:
               img = session.image
               # do something with image


We would need to think about adding metadata such as behavioral data
from the scanning session, and so on.  I suppose this will help us
move transparently to using something like HDF5 for data storage.
