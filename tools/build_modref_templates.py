#!/usr/bin/env python
"""Attempt to generate templates for module reference with Sphinx

XXX - the titles of the chapters are enormous, needs fixing.

XXX - we exclude extension modules

NOTE: this is a modified version of a script originally shipped with
the PyMVPA project, which we've adapted for NIPY use.  PyMVPA and NIPY
are both BSD-licensed projects.
"""

# Stdlib imports
import os

# Functions and classes
class ApiDocWriter(object):
    ''' Class for automatic detection and parsing of API docs
    to Sphinx-parsable reST format'''

    # only separating first two levels
    rst_section_levels = ['*', '=', '-', '~', '^']

    def __init__(self,
                 package_name,
                 outpath = None, # can be None for testing
                 exclude_list=None,
                 rst_extension='.rst',
                 package_skip_patterns=None,
                 fnames_to_skip=None):
        ''' Initialize package for parsing

        *package_name* must be importable

          package : string
            Name of the top-level package.

          modref_path : tuple
            Path components leading to the directory where the sources should be
            written.

        '''
        if exclude_list is None: exclude_list = []
        if package_skip_patterns is None: package_skip_patterns = []
        if fnames_to_skip is None: fnames_to_skip = ['setup.py']
        self.package_name = package_name
        self.root_module = __import__(package_name)
        self.root_path = self.root_module.__path__[0]
        self.outpath = outpath
        self.exclude_list = exclude_list
        self.rst_extension = rst_extension
        self.package_skip_patterns=package_skip_patterns
        self.fnames_to_skip = fnames_to_skip
        self.written_modules = None
            
    def _get_object_name(self, line):
        ''' Get second token in line
        >>> docwriter = ApiDocWriter('sphinx')
        >>> docwriter._get_object_name("  def func():  ")
        'func'
        >>> docwriter._get_object_name("  class Klass(object):  ")
        'Klass'
        >>> docwriter._get_object_name("  class Klass:  ")
        'Klass'
        '''
        name = line.split()[1].split('(')[0].strip()
        # in case we have classes which are not derived from object
        # ie. old style classes
        return name.rstrip(':')

    def _uri2path(self, uri):
        ''' Convert uri to absolute filepath
        >>> docwriter = ApiDocWriter('sphinx')
        >>> import sphinx
        >>> modpath = sphinx.__path__[0]
        >>> res = docwriter._uri2path('sphinx.modulename')
        >>> res == modpath + os.path.sep + 'modulename'
        True
        >>> res = docwriter._uri2path('sphinx')
        >>> res == modpath
        True
        '''
        if uri == self.package_name:
            return self.root_path
        path = uri.replace('.', os.path.sep)
        path = path.replace(self.package_name + os.path.sep, '')
        return os.path.join(self.root_path, path)

    def _path2uri(self, dirpath):
        ''' Convert directory path to uri '''
        relpath = dirpath.replace(self.root_path, self.package_name)
        if relpath.startswith(os.path.sep):
            relpath = relpath[1:]
        return relpath.replace(os.path.sep, '.')

    def _parse_module(self, filename):
        # get file uri
        if os.path.exists(filename + '.py'):
            filename += '.py'
        elif  os.path.exists(os.path.join(filename, '__init__.py')):
            filename = os.path.join(filename, '__init__.py')
        else:
            # nothing that we could handle here.
            return ([],[])
        f = open(filename, 'rt')
        functions, classes = self._parse_lines(f)
        f.close()
        return functions, classes
    
    def _parse_lines(self, linesource):
        ''' Parse lines of text for functions and classes '''
        functions = []
        classes = []
        for line in linesource:
            if line.startswith('def ') and line.count('('):
                # exclude private stuff
                name = self._get_object_name(line)
                if not name.startswith('_'):
                    functions.append(name)
            elif line.startswith('class '):
                # exclude private stuff
                name = self._get_object_name(line)
                if not name.startswith('_'):
                    classes.append(name)
            else:
                pass
        functions.sort()
        classes.sort()
        return functions, classes

    def write_api_doc(self, uri):
        # get the names of all classes and functions
        if self.outpath is None:
            raise ValueError('outpath is not set')
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)
        filename = self._uri2path(uri)
        functions, classes = self._parse_module(filename)
        if not len(functions) and not len(classes):
            print 'WARNING: Empty -',uri  # dbg
            return False
        tf = open(os.path.join(self.outpath,
                               uri + rst_extension), 'wt')
        ad = '.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n'
        if len(classes):
            ad += 'Inheritance diagram for: %s\n\n' % uri
            ad += '.. inheritance-diagram:: %s \n' % uri
            ad += '   :parts: 2\n\n'
        title = ':mod:`' + uri + '`'
        ad += title + '\n' + self.rst_section_levels[1] * len(title)
        ad += '\n.. automodule:: ' + uri + '\n'
        ad += '\n.. currentmodule:: ' + uri + '\n'
        multi_class = len(classes) > 1
        multi_fx = len(functions) > 1
        if multi_class:
            ad += '\n' + 'Classes' + '\n' + \
                  self.rst_section_levels[2] * 7 + '\n'
        elif len(classes) and multi_fx:
            ad += '\n' + 'Class' + '\n' + \
                  self.rst_section_levels[2] * 5 + '\n'
        for c in classes:
            ad += '\n:class:`' + c + '`\n' \
                  + self.rst_section_levels[multi_class + 2 ] * \
                  (len(c)+9) + '\n\n'
            ad += '\n.. autoclass:: ' + c + '\n'
            # must NOT exclude from index to keep cross-refs working
            ad += '  :members:\n' \
                  '  :undoc-members:\n' \
                  '  :show-inheritance:\n'
            #      '  :noindex:\n\n'
        if multi_fx:
            ad += '\n' + 'Functions' + '\n' + \
                  self.rst_section_levels[2] * 9 + '\n\n'
        elif len(functions) and multi_class:
            ad += '\n' + 'Function' + '\n' + \
                  self.rst_section_levels[2] * 8 + '\n\n'
        for f in functions:
            # must NOT exclude from index to keep cross-refs working
            ad += '\n.. autofunction:: ' + uri + '.' + f + '\n\n'
        tf.write(ad)
        tf.close()
        return True  # success

    def write_api_docs(self):
        """Generate API reST files.

        Sets self.written_modules - list of written module rst files
        """
        # compose list of modules
        modules = []

        # raw directory parsing
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            # determine the importable location of the module
            module_uri = self._path2uri(dirpath)
            # Skip any name that contains one of the provided skip patterns,
            # typically used to avoid unittests in docs
            skip = False
            for pat in self.package_skip_patterns:
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
                if not filename.endswith('.py') \
                       or filename.startswith('_') \
                       or filename in fnames_to_skip:
                    continue
                modules.append('.'.join([module_uri, filename[:-3]]))
        # write the list
        written_modules = []
        for m in modules:
            if not m in exclude_list:
                got_written = self.write_api_doc(m)
                if got_written:
                    written_modules.append(m)
        self.written_modules = written_modules

    def write_index(self, path, rootpath):
        """Make a reST API index file from written files

        Parameters
        ----------
        path : string
            Filename to write index to
        rootpath : string
            path to which written filenames are relative
        """
        if self.written_modules is None:
            raise ValueError('No modules written')
        # Path written into index is relative to rootpath
        pth, fname = os.path.split(path)
        relpath = pth.replace(rootpath, '')
        if relpath.startswith(os.path.sep):
            relpath = relpath[1:]
        idx = open(path,'wt')
        w = idx.write
        w('.. AUTO-GENERATED FILE -- DO NOT EDIT!\n\n')
        w('.. toctree::\n\n')
        for f in self.written_modules:
            w('   %s/%s\n' % (relpath,f))
        idx.close()
        
    
#*****************************************************************************
if __name__ == '__main__':
    
    package = 'neuroimaging'
    rst_extension = '.rst'
    exclude_list = []
    package_skip_patterns = ['neuroimaging.externals',
                            'neuroimaging.fixes',
                            '.tests']
    fnames_to_skip = ['setup.py']
    outpath = os.path.join('api','generated')
    docwriter = ApiDocWriter(
        package,
        outpath,
        exclude_list,
        rst_extension,
        package_skip_patterns,
        fnames_to_skip)
    docwriter.write_api_docs()
    docwriter.write_index('api/generated/gen.rst', 'api')
    print '%d files written' % len(docwriter.written_modules)
