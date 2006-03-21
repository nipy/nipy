import test_image
import test_kernel
import test_pipes
import unittest

import neuroimaging.image.formats.tests as test_formats

def suite():
    return unittest.TestSuite([test_kernel.suite(),
                               test_image.suite(),
                               test_pipes.suite(),
                               test_formats.suite()])
