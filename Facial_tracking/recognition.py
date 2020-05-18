import cv2
import numpy as np
import os, os.path
import logic.logconfig as log
from logic.clasify_known_faces import load_image_from_path, find_facial_encodings, find_face_locations
import click
import configparser


def execute_recognition():
    recognition_tolerance = 0.6
    frame = 3
    font = 2

    cam = cv2.VideoCapture(0)

    known_faces_list, known_names_list = loading_known_faces()

    print('Processing unknown faces...')
    while True:
        ret, image = cam.read()

        locations = find_face_locations(image)
        encodings = find_facial_encodings(image, locations)

        print(f', found {len(encodings)} faces')
        for facial_encoding, face_location in zip(encodings, locations):
            recognition_results = face_comparison_list(known_faces_list, facial_encoding, recognition_tolerance)
            facial_match = None
            if True in recognition_results:
                facial_match = known_names_list[recognition_results.index(True)]
                print(f' - {facial_match}')

                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                name_color = convert_name_to_color(facial_match)

                cv2.rectangle(image, top_left, bottom_right, name_color, frame)

                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                cv2.rectangle(image, top_left, bottom_right, name_color, cv2.FILLED)

                cv2.putText(image, facial_match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font)

        cv2.imshow('Face recognition', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def loading_known_faces():
    conf = configparser.ConfigParser()
    conf.read("./settings/configuration.ini")
    frecog_conf = conf["FACE_RECOGNITION"]

    print('Loading known faces and names...')
    known_faces_list = []
    known_names_list = []

    with click.progressbar(os.listdir(frecog_conf["KnownFacesDir"])) as faces:
        for name in faces:
            print(len(os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}")))
            for filename in os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}"):
                image = load_image_from_path(f"{frecog_conf['KnownFacesDir']}/{name}/{filename}")
                encoding = find_facial_encodings(image)
                if len(encoding) > 0:
                    encoding = encoding[0]
                else:
                    print("No faces found in the image!")
                    pass
                known_faces_list.append(encoding)
                known_names_list.append(name)

    return known_faces_list, known_names_list


def face_comparison_list(known_face_encodings, face_encoding_to_check, recognition_tolerance):
    return list(linear_face_distance(known_face_encodings, face_encoding_to_check) <= recognition_tolerance)


def linear_face_distance(face_encodings_list, face_to_compare):
    if len(face_encodings_list) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings_list - face_to_compare, axis=1)


def convert_name_to_color(name):
    name_color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return name_color