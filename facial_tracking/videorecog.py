import cv2
import numpy as np
import os, os.path
import logic.logconfig as log
from logic.classify_known_faces import load_image_from_path, find_facial_encodings, find_face_locations, find_raw_facial_landmarks
import click
import configparser
from logic.write_to_csv import csv_writer
from imutils import face_utils
import keyboard


logger = log.logger

show_facial_landmarks = False
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def execute_videorecog(model="large", path =' '):
    global show_facial_landmarks

    recognition_tolerance = 0.6
    frame = 3
    font = 2
    videoname = path 
    if(videoname == ' '):
        cam = cv2.VideoCapture('./vids/test.mp4')
    else:
        cam = cv2.VideoCapture(videoname)
    known_faces_list, known_names_list = loading_known_faces(model)
    cam.set(3, 426)
    cam.set(4, 240)
    logger.info("Processing unknown faces...")
    print('Processing unknown faces...')

    while True:
        if(cam.isOpened()):
            ret, image = cam.read()
            if ret:
                image = rescale_frame(image, percent=40)

                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                locations = find_face_locations(image)
                encodings = find_facial_encodings(image, locations,1,model)

                logger.info(f", found {len(encodings)} faces")
                print(f', found {len(encodings)} faces')

                for facial_encoding, face_location in zip(encodings, locations):
                    recognition_results = face_comparison_list(known_faces_list, facial_encoding, recognition_tolerance)

                    facial_match = None
                    if True in recognition_results:
                        facial_match = known_names_list[recognition_results.index(True)]

                        logger.info(f"Match found: {facial_match}")
                        print(f' - {facial_match}')

                        top_left = (face_location[3], face_location[0])
                        bottom_right = (face_location[1], face_location[2])

                        name_color = convert_name_to_color(facial_match)

                        cv2.rectangle(image, top_left, bottom_right, name_color, frame)

                        if show_facial_landmarks == True:
                            for face_land in find_raw_facial_landmarks(image,None, model):
                                for (x, y) in face_utils.shape_to_np(face_land):
                                    cv2.circle(image,(x, y),3,name_color,-1)

                        top_left = (face_location[3], face_location[2])
                        bottom_right = (face_location[1], face_location[2] + 22)

                        cv2.rectangle(image, top_left, bottom_right, name_color, cv2.FILLED)

                        cv2.putText(image, facial_match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), font)

                cv2.imshow('Face recognizing on video', image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if keyboard.is_pressed('f'):
                    if show_facial_landmarks == True:
                        show_facial_landmarks = False
                    else:
                        show_facial_landmarks = True
            else:
                break

    cam.release()
    cv2.destroyAllWindows()


def loading_known_faces(model):
    conf = configparser.ConfigParser()
    conf.read("./settings/configuration.ini")
    frecog_conf = conf["FACE_RECOGNITION"]

    logger.info("loading known faces and names...\n")
    print('Loading known faces and names...')

    known_faces_list = []
    known_names_list = []

    with click.progressbar(os.listdir(frecog_conf["KnownFacesDir"])) as faces:
        for name in faces:

            logger.info(len(os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}")))
            print(len(os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}")))

            for filename in os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}"):
                image = load_image_from_path(f"{frecog_conf['KnownFacesDir']}/{name}/{filename}")
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                encoding = find_facial_encodings(image,None,1,model)
                if len(encoding) > 0:
                    encoding = encoding[0]
                else:
                    logger.info("No faces found in the image!")
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
    linarg = np.linalg.norm(face_encodings_list - face_to_compare, axis=1)

    logger.info(f"Linarg norm value: {min(linarg)}")
    print("Linarg norm value: ",min(linarg))

    return linarg


def convert_name_to_color(name):
    name_color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return name_color