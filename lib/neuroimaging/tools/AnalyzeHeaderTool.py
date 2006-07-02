import sys
from optparse import OptionParser, Option

from neuroimaging.data import DataSource
from neuroimaging.refactoring.analyze import AnalyzeHeader



class AnalyzeHeaderTool (OptionParser):
    "Command-line tool for getting and setting Analyze header values."
    
    _usage= "%prog [options] <hdrfile>\n"+__doc__
    options = (
      Option('-a', '--attribute', dest="attname",
        help="Get or set this attribute"),
      Option('-v', '--value', dest="value",
        help="Set attribute to this value"))


    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.set_usage(self._usage)
        self.add_options(self.options)


    def _error(self, message):
        print message
        self.print_help()
        sys.exit(0)


    def run(self):
        options, args = self.parse_args()
        if len(args) != 1: self._error("Please provide a header file name")
        filename = args[0]
        if not DataSource().exists(filename):
            self._error("File not found: %s"%filename)
        header = AnalyzeHeader(filename)
        attname, value = options.attname, options.value
        if attname is not None:
            if value is not None:
                print "before: %s\t%s"%(attname, getattr(header, attname))
                setattr(header, attname, value)
                print "after: %s\t%s"%(attname, getattr(header, attname))
                header.write(filename+".new")
            else: print "%s\t%s"%(attname, getattr(header, attname))
        elif value is not None:
            self._error("Only provide a value when an attribute is provided")
        else: print header

if __name__ == "__main__": AnalyzeHeaderTool().run()
