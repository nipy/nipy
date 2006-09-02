import unittest

class TemplateTest(unittest.TestCase):

    def setUp(self):
        print "TestCase initialization..."

    def test_foo(self):
        print "testing foo"

    def test_bar(self):
        print "testing bar"
      

if __name__ == '__main__':
    unittest.main()
