"""
Module containing simple utilities for running our tests.
"""
import doctest

# This is a module level variable to keep track of which flags are set.
# It replaces the use of sys.argv we previously had. Perhaps we can find a
# way to do it with no global state at all... -- Tim

FLAGS = []

def set_flags(flags):
    from neuroimaging.utils.testutils import FLAGS
    for flag in ["slow", "gui", "data", "all"]:
        if flag in FLAGS:
            FLAGS.remove(flag)

    if type(flags) == str:
        FLAGS.append(flags)
    else:
        for flag in flags:
            FLAGS.append(flag)


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
                    if flag not in FLAGS and "all" not in FLAGS:
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

