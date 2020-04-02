import click
from logic import print_me

# pip install --editable .
# AFTER pipenv shell

@click.group()
def frecog():
    pass

@frecog.command()
@click.option("--test", "-t")
@click.option("--sync", "-s", is_flag=True)
def ex(test, sync):
    if(test):
        print_me(test)
    elif(sync):
        pass
    
    