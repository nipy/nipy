#!/usr/bin/env python
"""Attempt to generate templates for module reference with Sphinx

XXX - We still need to add the auto-generation of inheritance diagram
directives in the right places, so all modules get diagrams for their classes.

XXX - the titles of the chapters are enormous, needs fixing.

XXX - currently, we had to leave a symlink in the doc directory to the actual
package (doc/neuroimaging --> ../neuroimaging).  Instead, the
writeAPIDocTemplate routine should take the proper path info.


NOTE: this is a modified version of a script originally shipped with the PyMVPA
project, which we've adapted for NIPY use.  PyMVPA and NIPY are both
BSD-licensed projects.
"""

# Stdlib imports
import os
import re

# Globals
# only separating first two levels
rst_section_levels = ['*', '=', '-', '~', '^']

# Functions and classes
def getObjectName(line):
    name = line.split()[1].split('(')[0].strip()
    # in case we have classes which are niot derived from object
    # ie. old style classes
    return name.rstrip(':')


def parseModule(uri):
    filename = re.sub('\.', os.path.sep, uri)
    #print '*** uri,file:',uri,filename  # dbg

    # get file uri
    if os.path.exists(filename + '.py'):
        filename += '.py'
    elif  os.path.exists(os.path.join(filename, '__init__.py')):
        filename = os.path.join(filename, '__init__.py')
    else:
        # nothing that we could handle here.
        return ([],[])

    f = open(filename)

    functions = []
    classes = []

    for line in f:
        if line.startswith('def ') and line.count('('):
            # exclude private stuff
            name = getObjectName(line)
            if not name.startswith('_'):
                functions.append(name)
        elif line.startswith('class '):
            # exclude private stuff
            name = getObjectName(line)
            if not name.startswith('_'):
                classes.append(name)
        else:
            pass

    f.close()

    functions.sort()
    classes.sort()

    return functions, classes


def writeAPIDocTemplate(uri,modref_path,rst_extension='.rst'):
    # get the names of all classes and functions
    functions, classes = parseModule(uri)

    # do nothing if there is nothing to do
    if not len(functions) and not len(classes):
        print 'WARNING: Empty -',uri  # dbg
        return False

    tf = open(os.path.join(modref_path, uri + rst_extension), 'w')

    ad = '.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n'

    if len(classes) > 0:
        ad += 'Inheritance diagram for: %s\n\n' % uri
        ad += '.. inheritance-diagram:: %s \n' % uri
        ad += '   :parts: 2\n\n'
    
    title = ':mod:`' + uri + '`'
    ad += title + '\n' + rst_section_levels[1] * len(title)

    ad += '\n.. automodule:: ' + uri + '\n'
    ad += '\n.. currentmodule:: ' + uri + '\n'

    multi_class = len(classes) > 1
    multi_fx = len(functions) > 1
    if multi_class:
        ad += '\n' + 'Classes' + '\n' + rst_section_levels[2] * 7 + '\n'
    elif len(classes) and multi_fx:
        ad += '\n' + 'Class' + '\n' + rst_section_levels[2] * 5 + '\n'

    for c in classes:
        ad += '\n:class:`' + c + '`\n' \
              + rst_section_levels[multi_class + 2 ] * (len(c)+9) + '\n\n'

        ad += '\n.. autoclass:: ' + c + '\n'

        # must NOT exclude from index to keep cross-refs working
        ad += '  :members:\n' \
              '  :undoc-members:\n' \
              '  :show-inheritance:\n'
        #      '  :noindex:\n\n'


    if multi_fx:
        ad += '\n' + 'Functions' + '\n' + rst_section_levels[2] * 9 + '\n\n'
    elif len(functions) and multi_class:
        ad += '\n' + 'Function' + '\n' + rst_section_levels[2] * 8 + '\n\n'

    for f in functions:
        # must NOT exclude from index to keep cross-refs working
        ad += '\n.. autofunction:: ' + uri + '.' + f + '\n\n'

    tf.write(ad)
    tf.close()
    return True  # success


def gen_api_files(package,modref_path,exclude_list=None,rst_extension='.rst',
                  package_skip_patterns=None,fnames_to_skip=None):
    """Generate API reST files.

    Parameters
    ----------

      package : string
        Name of the top-level package.

      modref_path : tuple
        Path components leading to the directory where the sources should be
        written.

     Returns
     -------
       w : list
         The list of modules whose api files were written to disk.
      """

    # Default arguments
    
    if exclude_list is None: exclude_list = []
    if package_skip_patterns is None: package_skip_patterns = []
    if fnames_to_skip is None: fnames_to_skip = []
    
    # Code begins
    root_module = __import__(package)
    root_path = root_module.__path__[0]

    if not os.path.exists(modref_path):
        os.mkdir(modref_path)

    # compose list of modules
    modules = []

    # raw directory parsing
    for dirpath, dirnames, filenames in os.walk(root_path):
        # determine the importable location of the module
        module_uri = re.sub(os.path.sep,
                            '.',
                            re.sub(root_path,package,dirpath))
        print 'mod uri',module_uri
        print 'filenames:',filenames
        
        # Skip any name that contains one of the provided skip patterns,
        # typically used to avoid unittests in docs
        skip = False
        for pat in package_skip_patterns:
            #print 'checking pat:',pat,'uri:',module_uri  # dbg
            if pat in module_uri:
                skip = True
                break
        if skip:
            continue

        # no private module
        if not module_uri.count('._'):
            modules.append(module_uri)

        for filename in filenames:
            # XXX maybe check for extensions as well?
            # not private stuff
            if not filename.endswith('.py') or filename.startswith('_') \
                   or filename in fnames_to_skip:
                continue

            modules.append('.'.join([module_uri, filename[:-3]]))

    # write the list
    written_modules = []
    for m in modules:
        if not m in exclude_list:
            got_written = writeAPIDocTemplate(m,modref_path)
            if got_written:
                written_modules.append(m)
    return(written_modules)


def make_index(path,base_path,files):
    """Make a reST API index file for a given set of files.
    """
    idx = open(path,'wt')
    w = idx.write
    w('.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n')
    w('.. toctree::\n\n')
    for f in files:
        w('   %s/%s\n' % (base_path,f))
    idx.close()

    
#*****************************************************************************
if __name__ == '__main__':
    
    package = 'neuroimaging'
    rst_extension = '.rst'
    exclude_list = []
    package_skip_patterns = ['neuroimaging.externals',
                            'neuroimaging.fixes',
                            '.tests','.setup']
    fnames_to_skip = ['setup.py']
    modref_path = os.path.join('api','generated')
    written = gen_api_files(package,modref_path,exclude_list,rst_extension,
                            package_skip_patterns,fnames_to_skip)
    make_index(os.path.join('api','generated','gen.rst'),'generated',
               written)
