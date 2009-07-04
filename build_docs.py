"""
Code to build the documentation in the setup.py

To use this code, run::

    python setup.py build_sphinx
"""

# Standard library imports
import sys
import os
import zipfile
import warnings
import shutil
from distutils.cmd import Command
from distutils.command.clean import clean

# Sphinx import.
from sphinx.setup_command import BuildDoc

DOC_BUILD_DIR = os.path.join('build', 'html')


################################################################################
# Code for API generation 
class APIDocs(Command):
    description = \
    """ Generate API docs """

    user_options = [
        ('None', None, 'this command has no options'),
        ]

    def run(self):
        # First build and install nipy in a temporary location
        TEMP_INSTALL = os.path.join('build', 'install')
        install = self.distribution.get_command_obj('install')
        install.install_scripts = os.path.join('build', 'scripts')
        install.install_base    = TEMP_INSTALL 
        install.install_platlib = TEMP_INSTALL
        install.install_purelib = TEMP_INSTALL
        install.install_data    = TEMP_INSTALL
        install.install_lib     = TEMP_INSTALL
        install.install_headers = TEMP_INSTALL
        install.run()

        # Horrible trick to reload nipy with our temporary instal
        for key in sys.modules.keys():
            if key.startswith('nipy'):
                sys.modules.pop(key, None)
        sys.path.append(os.path.abspath(TEMP_INSTALL))
        # Pop the cwd
        sys.path.pop(0)
        import nipy

        os.chdir('doc')
        try:
            os.system("""%s -c 'import sys; sys.path.append("%s"); sys.path.append("%s"); execfile("../tools/build_modref_templates.py", dict(__name__="__main__"))'"""
                                % (sys.executable, 
                                   os.path.abspath('../tools'),
                                   os.path.abspath(TEMP_INSTALL)))
        finally:
            os.chdir('..')

    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass


################################################################################
# Code to copy the sphinx-generated html docs in the distribution.
def relative_path(filename):
    """ Return the relative path to the file, assuming the file is
        in the DOC_BUILD_DIR directory.
    """
    length = len(os.path.abspath(DOC_BUILD_DIR)) + 1
    return os.path.abspath(filename)[length:]


################################################################################
# Code to build the docs 
class MyBuildDoc(BuildDoc):
    """ Sub-class the standard sphinx documentation building system, to
        add logics for API generation and matplotlib's plot directive.
    """

    def run(self):
        self.run_command('api_docs')
        # We need to be in the doc directory for to plot_directive
        # and API generation to work
        os.chdir('doc')
        try:
            BuildDoc.run(self)
        finally:
            os.chdir('..')
        self.zip_docs()
    
    def zip_docs(self):
        if not os.path.exists(DOC_BUILD_DIR):
            raise OSError, 'Doc directory does not exist.'
        target_file = os.path.join('doc', 'documentation.zip')
        # ZIP_DEFLATED actually compresses the archive. However, there
        # will be a RuntimeError if zlib is not installed, so we check
        # for it. ZIP_STORED produces an uncompressed zip, but does not
        # require zlib.
        try:
            zf = zipfile.ZipFile(target_file, 'w', 
                                        compression=zipfile.ZIP_DEFLATED)
        except RuntimeError:
            warnings.warn('zlib not installed, storing the docs '
                            'without compression')
            zf = zipfile.ZipFile(target_file, 'w', 
                                        compression=zipfile.ZIP_STORED)    

        for root, dirs, files in os.walk(DOC_BUILD_DIR):
            relative = relative_path(root)
            if not relative.startswith('.doctrees'):
                for f in files:
                    zf.write(os.path.join(root, f), 
                            os.path.join(relative, 'html_docs', f))
        zf.close()


    def finalize_options(self):
        """ Override the default for the documentation build
            directory.
        """
        self.build_dir = os.path.join(*DOC_BUILD_DIR.split(os.sep)[:-1])
        BuildDoc.finalize_options(self)


################################################################################
# Code to clean
class Clean(clean):

    def run(self):
        clean.run(self)
        api_path = os.path.join('doc', 'api', 'generated')
        if os.path.exists(api_path):
            print "Removing %s" % api_path
            shutil.rmtree(api_path)
        if os.path.exists(DOC_BUILD_DIR):
            print "Removing %s" % DOC_BUILD_DIR 
            shutil.rmtree(DOC_BUILD_DIR)

# The command classes for distutils, used by the setup.py
cmdclass = {'build_sphinx': MyBuildDoc,
            'api_docs': APIDocs,
            'clean': Clean,
            }


