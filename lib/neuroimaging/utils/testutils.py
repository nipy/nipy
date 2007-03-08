"""
Module containing simple utilities for running our tests.
"""
import doctest, sys

class MyDocTestFinder(doctest.DocTestFinder):
    def find(self, obj, name=None, module=None, globs=None,
             extraglobs=None):
        results = doctest.DocTestFinder.find(self, obj, name, module, globs,
                                             extraglobs)
        for res in results:
            for ex in res.examples:
                flags = []
                options = ["slow", "gui", "data"]
                for opt in options:
                    if ex.source == "%s = True\n" % opt.upper():
                        flags.append("%s" % opt)                    
                for flag in flags:
                    if flag not in sys.argv and "all" not in sys.argv:
                        res.examples = []
                if res.examples == []:
                    break
        return results

def make_doctest_suite(module):
    def test_suite(level=1):
        import doctest, neuroimaging        
        m1 = __import__(module)
        return doctest.DocTestSuite(eval(module), test_finder=MyDocTestFinder())
    return test_suite

