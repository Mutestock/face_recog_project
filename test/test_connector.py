import unittest
from settings.connector import DB_connection

class Tests(unittest.TestCase):
    def test_connection(self):
        self.assertIsNotNone(DB_connection())

