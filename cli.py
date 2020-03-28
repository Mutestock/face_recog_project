import click
from logic import print_me

@click.group()
def manager():
    pass

@manager.command()
@click.option("--test", "-t")
@click.option("--sync", "-s", is_flag=True)
def ex(test, sync):
    if(test):
        print_me(test)
    elif(sync):
        pass
    
    