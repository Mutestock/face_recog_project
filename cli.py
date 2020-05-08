import click
from logic.recognition import face_recognition_load
from Facial_tracking.OpenCV_Facial import executor
from logic.video_handling import play_mp4
# pip install --editable .
# AFTER pipenv shell


@click.group()
def frecog():
    pass


@frecog.command()
@click.option("--iterate", "-i", is_flag=True)
@click.option("--find", "-f", is_flag=True)
def run(iterate, find):
    '''
    Primary executable functionalities
    '''
    if iterate:
        face_recognition_load()
    if(find):
        executor()
    

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
@click.option(
    '--succes', 
    '-s', 
    is_flag=True, 
    help='Shows the possible displayable entries for success rates graphs')
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
