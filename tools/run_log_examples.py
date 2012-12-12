#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, with_statement

DESCRIP = 'Run and log examples'
EPILOG = \
""" Run examples in directory

Typical usage is:

run_log_examples.py nipy/examples --log-path=~/tmp/eg_logs

to run the examples and log the result, or

run_log_examples.py nipy/examples/some_example.py

to run a single example.
"""

import sys
import os
from os.path import (abspath, expanduser, join as pjoin, sep as psep, isfile,
                     dirname)
from subprocess import Popen, PIPE
import re

from nibabel.py3k import asstr

from nipy.externals.argparse import (ArgumentParser,
                                     RawDescriptionHelpFormatter)


PYTHON=sys.executable
NEED_SHELL = True

class ProcLogger(object):
    def __init__(self, log_path, working_path):
        self.log_path = log_path
        self.working_path = working_path
        self._names = []

    def cmd_str_maker(self, cmd, args):
        return " ".join([cmd] + list(args))

    def __call__(self, cmd_name, cmd, args=(), cwd=None):
        # Mqke log files
        if cmd_name in self._names:
            raise ValueError('Command name {0} not unique'.format(cmd_name))
        self._names.append(cmd_name)
        if cwd is None:
            cwd = self.working_path
        cmd_out_path = pjoin(self.log_path, cmd_name)
        stdout_log = open(cmd_out_path + '.stdout', 'wt')
        stderr_log = open(cmd_out_path + '.stderr', 'wt')
        try:
            # Start subprocess
            cmd_str = self.cmd_str_maker(cmd, args)
            proc = Popen(cmd_str,
                        cwd = cwd,
                        stdout = stdout_log,
                        stderr = stderr_log,
                        shell = NEED_SHELL)
            # Execute
            retcode = proc.wait()
        finally:
            if proc.poll() is None: # In case we get killed
                proc.terminate()
            stdout_log.close()
            stderr_log.close()
        return retcode

    def run_pipes(self, cmd, args=(), cwd=None):
        if cwd is None:
            cwd = self.working_path
        try:
            # Start subprocess
            cmd_str = self.cmd_str_maker(cmd, args)
            proc = Popen(cmd_str,
                         cwd = cwd,
                         stdout = PIPE,
                         stderr = PIPE,
                         shell = NEED_SHELL)
            # Execute
            stdout, stderr = proc.communicate()
        finally:
            if proc.poll() is None: # In case we get killed
                proc.terminate()
        return asstr(stdout), asstr(stderr), proc.returncode


class PyProcLogger(ProcLogger):
    def cmd_str_maker(self, cmd, args):
        """ Execute python script `cmd`

        Reject any `args` because we're using ``exec`` to execute the script.

        Prepend some matplotlib setup to suppress figures
        """
        if len(args) != 0:
            raise ValueError("Cannot use args with {8}".format(self.__class__))
        return("""{0} -c "import matplotlib as mpl; mpl.use('agg'); """
               """exec(open('{1}', 'rt').read())" """.format(PYTHON, cmd))


def _record(result, fname, fileobj):
    print(result)
    fileobj.write('{0}: {1}\n'.format(fname, result))


def main():
    parser = ArgumentParser(description=DESCRIP,
                            epilog=EPILOG,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('examples_path', type=str,
                        help='filename of example or directory containing '
                             'examples to run')
    parser.add_argument('--log-path', type=str, default='',
                        help='path for output logs (default is cwd)')
    parser.add_argument('--excludex', type=str, action='append', default=[],
                        help='regex for files to exclude (add more than one '
                        '--excludex option for more than one regex filter')
    args = parser.parse_args()
    # Proc runner
    eg_path = abspath(expanduser(args.examples_path))
    if args.log_path == '':
        log_path = abspath(os.getcwd())
    else:
        log_path = abspath(expanduser(args.log_path))
    excludexes = [re.compile(s) for s in args.excludex]
    if isfile(eg_path): # example was a file
        proc_logger = PyProcLogger(log_path=log_path,
                                   working_path=dirname(eg_path))
        print("Running " + eg_path)
        stdout, stderr, code = proc_logger.run_pipes(eg_path)
        print('==== Stdout ====')
        print(stdout)
        print('==== Stderr ====')
        print(stderr)
        sys.exit(code)
    # Multi-run with logging to file
    proc_logger = PyProcLogger(log_path=log_path,
                               working_path=eg_path)
    fails = 0
    with open(pjoin(log_path, 'summary.txt'), 'wt') as f:
        for dirpath, dirnames, filenames in os.walk(eg_path):
            for fname in filenames:
                full_fname = pjoin(dirpath, fname)
                if fname.endswith(".py"):
                    print(fname, end=': ')
                    sys.stdout.flush()
                    for excludex in excludexes:
                        if excludex.search(fname):
                            _record('SKIP', fname, f)
                            break
                    else: # run test
                        cmd_name = full_fname.replace(eg_path + psep, '')
                        cmd_name = cmd_name.replace(psep, '-')
                        code = proc_logger(cmd_name, full_fname, cwd=dirpath)
                        if code == 0:
                            _record('OK', fname, f)
                        else:
                            fails += 1
                            _record('FAIL', fname, f)
    sys.exit(fails if fails < 255 else 255)


if __name__ == '__main__':
    main()
