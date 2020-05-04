import click
from logic.recognition import face_recognition_load
from Facial_tracking.OpenCV_Facial import executor
# pip install --editable .
# AFTER pipenv shell


@click.group()
def frecog():
    pass


@frecog.command()
@click.option("--iterate", "-i", is_flag=True)
@click.option("--find", "-f", is_flag=True)
def run(iterate, find):
    if iterate:
        face_recognition_load()
    if(find):
        executor()
