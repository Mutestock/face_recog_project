import click
from logic.recognition import face_recognition_load

# pip install --editable .
# AFTER pipenv shell


@click.group()
def frecog():
    pass


@frecog.command()
@click.option("--iterate", "-i", is_flag=True)
def run(iterate):
    if iterate:
        face_recognition_load()
