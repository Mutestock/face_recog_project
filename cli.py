import click
from facial_tracking.facial_tracking import execute_tracking
from facial_tracking.recognition import execute_recognition
from logic.recognition_file import loadrecog
#from logic.video_handling import play_mp4
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
        execute_recognition(model=model)

#Done
@frecog.command()
@click.option("--train", "-tr", type=click.Choice(['small', 'large']))
@click.argument('value', required=False)
def trainer(train, value):
    '''
    Train facial classifier
    E.g: frecog train-facial-classifier -tr large 2
    '''
    if train:
        if value != None:
            train_classifier(number_neighbors=int(value), model=train)
        else:
            train_classifier(model=train)


#Done
@frecog.command()
@click.option('--path','-p')
@click.option("--single", "-s")
def classify(single, path):
    '''
    Classifies unknown pictures in a directory using the knn_model

    E.g. path:   frecog classify -p facerec/unknown_faces
    E.g. single: frecog classify -s C:/Users/rasmu/Desktop/face_recog_project/facerec/unknown_faces/eka01.jpg
    '''
    if path:
        classify_people_from_path(path)
    elif single:
        classify_single_image(single)

#Done
@frecog.command()
@click.option("--csv", "-c", nargs=2, required=True)
@click.option("--benchmark", '-b')
def graph(csv, benchmark):
    '''
    where csv[1] is name and csv[0] is file name
    Plots a graph of the linalg norm distance 
    Eks: frecog graph -c rasmusb1.csv Rasmus
    Eks benchmark: frecog graph -c rasmusb1.csv Rasmus -b large
    '''
    if csv:
        if benchmark:
            execute_recognition(model=benchmark, benchmark=csv[0])
        plot_csv_data(csv[1], csv[0])


@frecog.command()   
@click.option('--movie' , '-m', is_flag=True)
@click.argument('model', required=False)
@click.argument('path', required=False)
def play(movie, model, path):
    '''
    Face recognises on a video
    Eks: frecog play -m small ./vids/pathTest.mp4
    '''
    if(movie):
        execute_videorecog(model, path)

@frecog.command()
@click.option("--folder", "-f", is_flag=True)
@click.argument('path', required=False)
def fold(folder, path):
    '''
    Face recognises on a folder of pictures
    Eks: frecog fold -f ./facerec/unknown_faces
    '''
    if folder:
        loadrecog(path)
