import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import dlib
import numpy as np
from pkg_resources import resource_filename


model_save_path = os.path.join("face_learning_model/models/knn_model.clf")
train_dir = "facerec/known_faces"
model_path = os.path.join("face_learning_model/models/knn_model.clf")

def five_point_model_location():
    return os.path.join("face_learning_model/models/shape_predictor_5_face_landmarks.dat")

def sixty_eight_point_model_location():
    return os.path.join("face_learning_model/models/shape_predictor_68_face_landmarks.dat")

def recognition_model_location():
    return os.path.join("face_learning_model/models/dlib_face_recognition_resnet_model_v1.dat")

def cnn_model_location():
    return os.path.join("face_learning_model/models/mmod_human_face_detector.dat")

face_detector = dlib.get_frontal_face_detector()

five_point_model = five_point_model_location()
five_point_predictor = dlib.shape_predictor(five_point_model)

sixty_eight_point_model = sixty_eight_point_model_location()
sixty_eight_point_predictor = dlib.shape_predictor(sixty_eight_point_model)

face_recognition_model = recognition_model_location()
facial_encoder = dlib.face_recognition_model_v1(face_recognition_model)

cnn_face_detection_model = cnn_model_location()
cnn_facial_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)


def train_classifier(number_neighbors=None, knn_algorithm='ball_tree',model='small'):
    
    X_train = []
    y_train = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in find_images_in_folder(os.path.join(train_dir, class_dir)):
            image = load_image_from_path(img_path)
            face_bounding_boxes = find_face_locations(image)

            if len(face_bounding_boxes) != 1:
                print("No people in the picture: ", img_path)
            else:
                X_train.append(find_facial_encodings(image, face_bounding_boxes, 1, model)[0])
                y_train.append(class_dir)

    if number_neighbors is None:
        number_neighbors = int(round(math.sqrt(len(X_train))))
        print("Number of neighbors: ", number_neighbors)

    knn_clf_model = neighbors.KNeighborsClassifier(n_neighbors=number_neighbors, algorithm=knn_algorithm, weights='distance')
    knn_clf_model.fit(X_train, y_train)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf_model, f)

    return knn_clf_model


def find_images_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]


def predict(image, distance_threshold=0.6, model='small'):

    with open(model_path, 'rb') as f:
        knn_clf_model = pickle.load(f)

    face_locations = find_face_locations(image)

    if len(face_locations) == 0:
        return []

    facial_encodings = find_facial_encodings(image, face_locations, 1, model)

    closest_distances = knn_clf_model.kneighbors(facial_encodings, n_neighbors=1)
    matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    prediction = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf_model.predict(facial_encodings), face_locations, matches)]
    
    return prediction


def find_facial_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    raw_facial_landmarks = find_raw_facial_landmarks(face_image, known_face_locations, model)
    return [np.array(facial_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_facial_landmarks]


def find_raw_facial_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = find_raw_face_locations(face_image)
    else:
        face_locations = [location_rectangle(face_location) for face_location in face_locations]

    pose_predictor = sixty_eight_point_predictor

    if model == "small":
        pose_predictor = five_point_predictor

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def location_rectangle(dimensions):
    return dlib.rectangle(dimensions[3], dimensions[0], dimensions[1], dimensions[2])


def find_face_locations(image, number_upsample=1, model="hog"):
    if model == "cnn":
        return [trim_rectangle_dimensions(rectangle_dimensions(face.rect), image.shape) for face in find_raw_face_locations(image, number_upsample, "cnn")]
    else:
        return [trim_rectangle_dimensions(rectangle_dimensions(face), image.shape) for face in find_raw_face_locations(image, number_upsample, model)]


def find_raw_face_locations(image, number_upsample=1, model="hog"):
    if model == "cnn":
        return cnn_facial_detector(image, number_upsample)
    else:
        return face_detector(image, number_upsample)


def rectangle_dimensions(rectangle):
    return rectangle.top(), rectangle.right(), rectangle.bottom(), rectangle.left()


def trim_rectangle_dimensions(rectangle_dimension, image_shape):
    return max(rectangle_dimension[0], 0), min(rectangle_dimension[1], image_shape[1]), min(rectangle_dimension[2], image_shape[0]), max(rectangle_dimension[3], 0)


def load_image_from_path(file, mode='RGB'):
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def classify_people_from_path(picture_path):
    for image_file in os.listdir(picture_path):
        full_file_path = os.path.join(picture_path, image_file)

        print("Looking for faces in {}".format(full_file_path))
        image = load_image_from_path(full_file_path)
        predictions = predict(image)

        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))


def classify_single_image(full_picture_path):
    image = load_image_from_path(full_picture_path)
    predictions = predict(image)

    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))