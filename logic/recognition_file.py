import cv2
import numpy as np
import os, os.path
import logic.logconfig as log
from logic.classify_known_faces import load_image_from_path, find_facial_encodings, find_face_locations
import click
import configparser
import keyboard


KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TESTFACE_DIR = 'testimage'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

conf = configparser.ConfigParser()
conf.read("./settings/configuration.ini")
frecog_conf = conf["FACE_RECOGNITION"]

#video = cv2.VideoCapture(0)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.5):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def face_distance(face_encodings_list, face_to_compare):
    if len(face_encodings_list) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings_list - face_to_compare, axis=1)

def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def loadrecog(path=' '):
    print('Loading known faces...')
    known_faces = []
    known_names = []
    file_path = path
    if path == None:
        file_path="./facerec/testfaces"

    with click.progressbar(os.listdir(frecog_conf["KnownFacesDir"])) as faces:
        for name in faces:
            print(len(os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}")))
            for filename in os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}"):
                image = load_image_from_path(
                    f"{frecog_conf['KnownFacesDir']}/{name}/{filename}"
                )
                encoding = find_facial_encodings(image)
                if len(encoding) > 0:
                    encoding = encoding[0]
                else:
                    print("No faces found in the image!")
                    pass
                known_faces.append(encoding)
                known_names.append(name)
    count =0
    print('Processing unknown faces...')
    while True:
        size = len(os.listdir(file_path))
        #"./facerec/testfaces"
        for name in os.listdir(file_path):
            image = load_image_from_path(f"{file_path}/{name}")
            #video.read()
            locations = find_face_locations(image)
            encodings = find_facial_encodings(image, locations)
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height = image.shape[0] 
            width = image.shape[1] 
            if height > 490 or width > 690:
                image = rescale_frame(image, 50)
            cv2.imshow(name, image)
            
            count+=1
            print(f'{count} out of: {size}')
            
            if cv2.waitKey(3000) & 0xFF == ord("q"):
                cv2.destroyWindow(name)
            else:
                cv2.destroyWindow(name)
                
            
        if count>=size:
            #cv2.destroyWindow(name)
            break
     #video.release()
    cv2.destroyAllWindows()