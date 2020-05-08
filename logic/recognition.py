import face_recognition
import os
import cv2
import configparser
import click
import logic.logconfig as log

logger = log.logger
video = cv2.VideoCapture(0)

# Clean up. Separate in multiple files annd functions - They're too big.

def face_recognition_load():
    conf = configparser.ConfigParser()
    conf.read("./settings/configuration.ini")

    frecog_conf = conf["FACE_RECOGNITION"]
    cv2_conf = conf["CV2"]

    known_faces = []
    known_names = []

    logger.info("loading known faces...\n")

    with click.progressbar(os.listdir(frecog_conf["KnownFacesDir"])) as faces:
        for name in faces:
            for filename in os.listdir(f"{frecog_conf['KnownFacesDir']}/{name}"):
                image = face_recognition.load_image_file(
                    f"{frecog_conf['KnownFacesDir']}/{name}/{filename}"
                )
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(name)
    __unknown_faces_processing(cv2_conf, frecog_conf, known_faces, known_names)
    #__unknown_faces_processing_video(cv2_conf, frecog_conf, known_faces, known_names) #Video mode


def __unknown_faces_processing(cv2_conf, frecog_conf, known_faces, known_names):
    logger.info("Processing unknown faces...")

    for filename in os.listdir(frecog_conf["UnknownFacesDir"]):
        logger.info(filename)
        image = face_recognition.load_image_file(
            f"{frecog_conf['UnknownFacesDir']}/{filename}"
        )
        locations = face_recognition.face_locations(image, model=cv2_conf["Model"])
        encodings = face_recognition.face_encodings(image, locations)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(
                known_faces, face_encoding, float(frecog_conf["Tolerance"])
            )
            match = None
            if True in results:
                match = known_names[results.index(True)]
                logger.info(f"Match found: {match}")
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = [0, 225, 0]
                cv2.rectangle(
                    image,
                    top_left,
                    bottom_right,
                    color,
                    int(cv2_conf["FrameThickness"]),
                )
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(
                    image,
                    match,
                    (face_location[3] + 10, face_location[2] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    int(cv2_conf["FontThickness"]),
                )

        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)


def __unknown_faces_processing_video(cv2_conf, frecog_conf, known_faces, known_names):
    logger.info("Video: Processing unknown faces...")

    while True:
        ret, image = video.read()  
        locations = face_recognition.face_locations(image, model=cv2_conf["Model"])
        encodings = face_recognition.face_encodings(image, locations)
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(
                known_faces, face_encoding, float(frecog_conf["Tolerance"])
            )
            match = None
            if True in results:
                match = known_names[results.index(True)]
                logger.info(f"Video: Match found: {match}")
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = [0, 225, 0]
                cv2.rectangle(
                    image,
                    top_left,
                    bottom_right,
                    color,
                    int(cv2_conf["FrameThickness"]),
                )
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(
                    image,
                    match,
                    (face_location[3] + 10, face_location[2] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    int(cv2_conf["FontThickness"]),
                )
        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow(filename)
            break
