import cv2
import os, os.path
from settings.pathing import os_parse_path
import time
import tkinter as tk
from tkinter import simpledialog
import keyboard
from logic.Clasify_known_faces import load_image_file, face_encodings, face_locations
import configparser
import logic.logconfig as log
import click
import numpy as np


USER_INP = " "
known_faces_path = "./facerec/known_faces"
  
cv2dir = os.path.dirname(cv2.__file__)

face_cascade_path = os_parse_path(f"{cv2dir}\data\haarcascade_frontalface_default.xml")
eye_cascade_path = os_parse_path(f"{cv2dir}\data\haarcascade_eye.xml")
smile_cascade_path = os_parse_path(f"{cv2dir}\data\haarcascade_smile.xml")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

ROOT = tk.Tk()
ROOT.withdraw()

cam = cv2.VideoCapture(0)

logger = log.logger

face_locations_list = []
face_encodings_list = []
face_names = []
process_this_frame = True

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def face_distance(face_encodings_list, face_to_compare):
    if len(face_encodings_list) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings_list - face_to_compare, axis=1)


conf = configparser.ConfigParser()
conf.read("./settings/configuration.ini")

frecog_conf = conf["FACE_RECOGNITION"]
cv2_conf = conf["CV2"]

known_face_encodings = []
known_face_names = []

logger.info("loading known faces...\n")

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

while True:
    _,frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    path, dirs, files = next(os.walk(known_faces_path))
    count = len(files)

    if process_this_frame:
        face_locations_list = face_locations(frame)
        face_encodings_list = face_encodings(frame, face_locations_list)

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
         
        if keyboard.is_pressed('r'):
            if count < 10:
                if (USER_INP == " "):
                    USER_INP = simpledialog.askstring(title="Name", prompt="What's your Name?:")
                pathFace = "facerec/known_faces/" + USER_INP + "/"
                if not os.path.exists(pathFace):
                    os.makedirs(pathFace)
                    known_faces_path = pathFace
                pic = pathFace +  USER_INP + str(count)+".jpg"
                print(pic)
                cv2.imwrite(pic, frame)
                count += 1

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        roi_gray = gray[top:bottom, left:right]
        roi_color = frame[top:bottom, left:right]   

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0, 255, 0), 1)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 25)
        for (xs,ys,ws,hs) in smile:
            cv2.rectangle(roi_color, (xs,ys), (xs+ws, ys+hs),(255, 0, 0), 1)

    cv2.imshow('Face recognition',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
            
cam.release()
cv2.destroyAllWindows()


if(__name__ == '__main__'):
    cvf2dirtest = os.path.dirname(cv2.__file__)
    face_cascade_pathtest = os_parse_path(f"{cvf2dirtest}\data\haarcascade_frontalface_default.xml")
    print(face_cascade_pathtest)