import click
from logic import face_recognition_import_test

# pip install --editable .
# AFTER pipenv shell

@click.group()
def frecog():
    pass

@frecog.command()
@click.option("--test", "-t", is_flag=True)
@click.option("--sync", "-s", is_flag=True)
def ex(test, sync):
    if(test):
        face_recognition_import_test()
    elif(sync):
        pass
    
    