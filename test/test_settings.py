import unittest
from settings.pathing import os_parse_path
import os
import cv2

class Tests(unittest.TestCase):
    
    def test_pathing(self):
        cv2dir = os.path.dirname(cv2.__file__)
        self.assertIsNotNone(os_parse_path(f"{cv2dir}\data\\haarcascade_fronta/face_default.xml"))

