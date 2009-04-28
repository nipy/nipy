version='0.1.2'
release=False

if not release:
    have_revno = False
    try:
        import os
        import subprocess
        # Use bzr to get the revision number.  Execute bzr in the same
        # directory as this file.
        dir = os.path.dirname(__file__)
        proc = subprocess.Popen(['bzr', 'revno'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                cwd=dir)
        revno, errcode = proc.communicate()
        if revno:
            revno = int(revno)
            version += 'dev%d' % revno
            have_revno = True
    except OSError:
        # Either bzr was not found or nipy was imported from a
        # directory that isn't a bzr branch.  Just ignore.
        pass
    finally:
        if not have_revno:
            # just tack on a dev label
            version += '.dev'
