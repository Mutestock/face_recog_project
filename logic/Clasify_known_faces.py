import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import dlib
import numpy as np
from pkg_resources import resource_filename

model_save_path = "C:/Users/rasmu/Desktop/face_recog_project/Face_learning_model/models/knn_model.clf"
train_dir = "facerec/known_faces"
model_path = "C:/Users/rasmu/Desktop/face_recog_project/Face_learning_model/models/knn_model.clf"

def pose_predictor_five_point_model_location():
    return "C:/Users/rasmu/Desktop/face_recog_project/Face_learning_model/models/shape_predictor_5_face_landmarks.dat"

def pose_predictor_model_location():
    return "C:/Users/rasmu/Desktop/face_recog_project/Face_learning_model/models/shape_predictor_68_face_landmarks.dat"

def face_recognition_model_location():
    return "C:/Users/rasmu/Desktop/face_recog_project/Face_learning_model/models/dlib_face_recognition_resnet_model_v1.dat"

def cnn_face_detector_model_location():
    return "C:/Users/rasmu/Desktop/face_recog_project/Face_learning_model/models/mmod_human_face_detector.dat"

face_detector = dlib.get_frontal_face_detector()

predictor_5_point_model = pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

predictor_68_point_model = pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

cnn_face_detection_model = cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def train(n_neighbors=None, knn_algo='ball_tree'):
    
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = load_image_file(img_path)
            face_bounding_boxes = face_locations(image)

            if 1 != len(face_bounding_boxes) != 1:
                print("No people in the picture: ", img_path)
            else:
                X.append(face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        print("Number of neighbors:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]


def predict(X_img_path, distance_threshold=0.6):

    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    X_img = load_image_file(X_img_path)
    X_face_locations = face_locations(X_img)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_encodings(X_img, known_face_locations=X_face_locations)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    
    return predictions


def show_prediction_labels_on_image(img_path, predictions):

    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        name = name.encode("UTF-8")

        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw

    pil_image.show()


def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def face_locations(img, number_of_times_to_upsample=1, model="hog"):
    if model == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]


def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)


def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def load_image_file(file, mode='RGB'):
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)