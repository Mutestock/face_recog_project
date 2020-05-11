import cv2
import numpy as np
import os, os.path
import logic.logconfig as log
from logic.Clasify_known_faces import load_image_file, face_encodings, face_locations
import click
import configparser

video_capture = cv2.VideoCapture(0)

conf = configparser.ConfigParser()
conf.read("./settings/configuration.ini")
frecog_conf = conf["FACE_RECOGNITION"]

known_face_encodings = []
known_face_names = []

logger = log.logger
logger.info("loading known faces...\n")


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def face_distance(face_encodings_list, face_to_compare):
    if len(face_encodings_list) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings_list - face_to_compare, axis=1)


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
            known_face_encodings.append(encoding)
            known_face_names.append(name)

face_locations_list = []
face_encodings_list = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame

    if process_this_frame:

        face_locations_list = face_locations(rgb_small_frame)
        face_encodings_list = face_encodings(rgb_small_frame, face_locations_list)

        face_names = []
        for face_encoding in face_encodings_list:

            matches = compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()