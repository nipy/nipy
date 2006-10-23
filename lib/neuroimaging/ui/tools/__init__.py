"Package containing implementations of command-line tools and applications."

import sys
from optparse import OptionParser, Option

class BaseTool(OptionParser):
    """
    This is an abstract base class for command-line tools to inherit
    from.  It should never be instantiated directly and subclasses
    should overload the run() method.
    """

    _usage = "%prog [options]\n"+__doc__

    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.set_usage(self._usage)
        self.add_options(self._options)


    def _error(self, message):
        """
        Prints error message and exits.
        """
        print '\n-----------------------------------'
        print 'ERROR MESSAGE:'
        print message
        print
        print 'For more information about using this command,'
        print 'please read the usage information below:'
        print '-----------------------------------\n'
        self.print_help()
        sys.exit(0)

    def run():
        raise NotImplementedError
