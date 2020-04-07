import unittest
from logic import print_me


class TestThing(unittest.TestCase):
    def test_hi(self):
        self.assertEqual(print_me("cake"), "cake")
