import cv2
import os, os.path
from settings.pathing import os_parse_path
import time
import tkinter as tk
from tkinter import simpledialog
import keyboard

USER_INP = ""
known_faces_path = "./facerec/known_faces"
take_picture = False

def executor():
    #Loading cascades    
    cv2dir = os.path.dirname(cv2.__file__)

    face_cascade_path = os_parse_path(f"{cv2dir}\data\haarcascade_frontalface_default.xml")
    eye_cascade_path = os_parse_path(f"{cv2dir}\data\haarcascade_eye.xml")
    smile_cascade_path = os_parse_path(f"{cv2dir}\data\haarcascade_smile.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

    ROOT = tk.Tk()
    ROOT.withdraw()

    #Detection function
    def detect(grey, frame):
        global known_faces_path
        global USER_INP
        global take_picture

        path, dirs, files = next(os.walk(known_faces_path))
        count = len(files)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]  

            if keyboard.is_pressed('r'):
                name = simpledialog.askstring(title="Name", prompt="What's your Name?:")
                if (name != None):
                    take_picture = True
                    USER_INP = name

            if take_picture == True:
                if count <= 5:
                    pathFace = "facerec/known_faces/" + USER_INP + "/"
                    if not os.path.exists(pathFace):
                        os.makedirs(pathFace)
                        known_faces_path = pathFace
                    pic = pathFace +  USER_INP + str(count)+".jpg"
                    cv2.imwrite(pic, frame)
                    count += 1
                else:
                    take_picture = False 
                    USER_INP == ""
                    known_faces_path = "facerec/known_faces/" 
                
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0, 255, 0), 1)

            smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 25)
            for (xs,ys,ws,hs) in smile:
                cv2.rectangle(roi_color, (xs,ys), (xs+ws, ys+hs),(255, 0, 0), 1)

        return frame


    #Face recognition with webcam
    cam = cv2.VideoCapture(0)
    while True:
        _,frame=cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray,frame)
        cv2.imshow('Face recognition',canvas)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
                
    cam.release()
    cv2.destroyAllWindows()