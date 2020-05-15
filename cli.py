import click
from facial_tracking.facial_tracking import execute_tracking
from facial_tracking.recognition import execute_recognition
from logic.video_handling import play_mp4
from logic.clasify_known_faces import train_classifier, classify_people_from_path, classify_single_image
# pip install --editable .
# AFTER pipenv shell


@click.group()
def frecog():
    pass

#Done
@frecog.command()
@click.option("--track", "-t", is_flag=True)
@click.option("--recognize", "-r", is_flag=True)
def run(track, recognize):
    '''
    Primary executable functionalities
    '''
    if track:
        execute_tracking()
    if recognize:
        execute_recognition()

#Done
@frecog.command()
@click.option("--train", "-tr", is_flag=True)
@click.argument('value', required=False)
def train_facial_classifier(train, value):
    '''
    Train facial classifier
    '''
    if train:
        if value != None:
            if value.isnumeric():
                train_classifier(int(value))
        else:
            train_classifier()

#Done
@frecog.command()
@click.option("--classify", "-c", is_flag=True)
@click.argument('path', required=True)
def classify_people(classify, path):
    '''
    Classifies unknown pictures in a directory using the knn_model
    path: eks. facerec/unknown_faces
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
    path: eks. C:/Users/rasmu/Desktop/face_recog_project/facerec/unknown_faces/eka01.jpg
    '''
    if single:
        classify_single_image(path)


@frecog.command()
@click.argument('url', nargs=-1)
def stream_detection(url):
    '''
    Opens any number of youtube links with the face recognition functions
    '''
    # Stretch 
    # Open selenium on links
    # Window capture
    # Face recognition function
    raise NotImplementedError()


@frecog.command()
@click.option('--succes', '-s', is_flag=True, help='Shows the possible displayable entries for success rates graphs')
def info(success):
    '''
    Shows user practical information 
    '''
    # Read entry names
    # format tpd.
    raise NotImplemented()


@frecog.command()
@click.option('--movie' , '-m', type=click.Choice(['matrix']))
def play(movie):
    if(movie=='matrix'):
        play_mp4("file:///home/mute/Downloads/The+Matrix+-+A+system+of+Control.mp4")
