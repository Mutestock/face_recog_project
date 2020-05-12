import cv2
import numpy as np
import os, os.path
import logic.logconfig as log
from logic.clasify_known_faces import load_image_file, face_encodings, face_locations
import click
import configparser


KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

conf = configparser.ConfigParser()
conf.read("./settings/configuration.ini")
frecog_conf = conf["FACE_RECOGNITION"]

video = cv2.VideoCapture(0)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.5):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def face_distance(face_encodings_list, face_to_compare):
    if len(face_encodings_list) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings_list - face_to_compare, axis=1)

def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading known faces...')
known_faces = []
known_names = []


with click.progressbar(os.listdir(frecog_conf["KnownFacesDir"])) as faces:
    for name in faces:
        print(len(os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}")))
        for filename in os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}"):
            image = load_image_file(
                f"{frecog_conf['KnownFacesDir']}/{name}/{filename}"
            )
            encoding = face_encodings(image)
            if len(encoding) > 0:
                encoding = encoding[0]
            else:
                print("No faces found in the image!")
                pass
            known_faces.append(encoding)
            known_names.append(name)


print('Processing unknown faces...')
while True:
    ret, image = video.read()

    locations = face_locations(image)
    encodings = face_encodings(image, locations)

    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        results = compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = name_to_color(match)

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow('Face recognition', image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()