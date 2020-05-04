import cv2

def executor():
    #Loading cascades

    face_cascade = cv2.CascadeClassifier('C:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Python/Lib/site-packages/cv2/data/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('C:/Python/Lib/site-packages/cv2/data/haarcascade_smile.xml')

    #Detection function

    def detect(grey, frame):
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

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