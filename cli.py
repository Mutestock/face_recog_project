import click
from facial_tracking.facial_tracking import execute_tracking
from facial_tracking.recognition import execute_recognition
from logic.video_handling import play_mp4
#from logic.recognition_file import loadrecog
from logic.classify_known_faces import train_classifier, classify_people_from_path, classify_single_image
from logic.write_to_csv import plot_csv_data
from facial_tracking.videorecog import execute_videorecog
# pip install --editable .
# AFTER pipenv shell


@click.group()
def frecog():
    pass

#Done
@frecog.command()
@click.option("--track", "-t", is_flag=True)
@click.option("--recognize", "-r", is_flag=True)
@click.argument('model', required=False)
def run(track, recognize, model):
    '''
    Primary executable functionalities
    Eks: frecog run -t
    Eks: frecog run -r
    '''
    if track:
        execute_tracking()
    if recognize:
        execute_recognition(model)

#Done
@frecog.command()
@click.option("--train", "-tr", is_flag=True)
@click.argument('value', required=False)
@click.argument('model', required=False)
def train_facial_classifier(train, value, model):
    '''
    Train facial classifier
    Eks: frecog train-facial-classifier -tr 2 large
    '''
    if train:
        if value != None:
            train_classifier(number_neighbors=int(value), model=model)
        else:
            train_classifier(model=model)

#Done
@frecog.command()
@click.option("--classify", "-c", is_flag=True)
@click.argument('path', required=True)
def classify_people(classify, path):
    '''
    Classifies unknown pictures in a directory using the knn_model
    Eks: frecog classify-people -c facerec/unknown_faces
    '''
    if classify:
        classify_people_from_path(path)

#Done
@frecog.command()
@click.option("--single", "-si", is_flag=True)
@click.argument('path', required=True)
def classify_single_person(single, path):
    '''
    Classifies unknown pictures in a directory using the knn_model
    Eks: frecog classify-single-person -si C:/Users/rasmu/Desktop/face_recog_project/facerec/unknown_faces/eka01.jpg
    '''
    if single:
        classify_single_image(path)

#Done
@frecog.command()
@click.option("--graph", "-g", is_flag=True)
def csv_to_graph(graph):
    '''
    Plots a graph of the linalg norm distance 
    Eks: frecog csv-to-graph -g
    '''
    if graph:
        plot_csv_data()


@frecog.command()
@click.option('--movie' , '-m', is_flag=True)
@click.argument('model', required=False)
def play(movie, model):
    if(movie):
        execute_videorecog(model)