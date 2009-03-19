version='0.1.2'
release=False

if not release:
    try:
        import subprocess
        proc = subprocess.Popen(['bzr', 'revno'], stdout=subprocess.PIPE)
        revno, errcode = proc.communicate()
        revno = int(revno)
        version += 'r%d' % revno
    except OSError:
        # bzr was probably not found
        # just tack on a dev label
        version += '.dev'
