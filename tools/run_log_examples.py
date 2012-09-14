#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, with_statement

DESCRIP = 'Run and log examples'
EPILOG = \
""" Run examples in directory

Typical usage is:

run_log_examples.py nipy/examples --log-path=~/tmp/eg_logs
"""

import sys
import os
from os.path import abspath, expanduser, join as pjoin, sep as psep
from subprocess import Popen
import re

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


def main():
    parser = ArgumentParser(description=DESCRIP,
                            epilog=EPILOG,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('examples_path', type=str,
                        help='directory containing examples')
    parser.add_argument('--log-path', type=str, required=True,
                        help='path for output logs')
    parser.add_argument('--excludex', type=str, action='append', default=[],
                        help='regex for files to exclude (add more than one '
                        '--excludex option for more than one regex filter')
    args = parser.parse_args()
    # Proc runner
    eg_path = abspath(expanduser(args.examples_path))
    log_path = abspath(expanduser(args.log_path))
    excludexes = [re.compile(s) for s in args.excludex]
    proc_logger = PyProcLogger(log_path=log_path,
                               working_path=eg_path)
    fails = 0
    with open(pjoin(log_path, 'summary.txt'), 'wt') as f:
        for dirpath, dirnames, filenames in os.walk(eg_path):
            for fname in filenames:
                if fname.endswith(".py"):
                    print(fname, end=': ')
                    sys.stdout.flush()
                    for excludex in excludexes:
                        if excludex.search(fname):
                            result_str = "SKIP"
                            fail = 0
                            break
                    else:
                        full_fname = pjoin(dirpath, fname)
                        cmd_name = full_fname.replace(eg_path + psep, '')
                        cmd_name = cmd_name.replace(psep, '-')
                        code = proc_logger(cmd_name, full_fname, cwd=dirpath)
                        fail = code != 0
                        result_str = "FAIL" if fail else "OK"
                    print(result_str)
                    f.write('{0}: {1}\n'.format(fname, result_str))
                    fails += fail
    exit(fails if fails < 255 else 255)


if __name__ == '__main__':
    main()
