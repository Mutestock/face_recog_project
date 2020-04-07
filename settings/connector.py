import pymysql
import configparser
import getpass
from logic.decorators import Decorators
from sqlalchemy import create_engine


@Decorators.determine_environment
def DB_connection():
    active_environment = get_active_environment()
    conf = configparser.ConfigParser()
    conf.read("./settings/sensitive.ini")
    cnx = pymysql.connect(
        user=conf[active_environment]["SQLUsername"],
        password=conf[active_environment]["SQLPassword"],
        host=conf[active_environment]["SQLHost"],
        port=int(conf[active_environment]["SQLPort"]),
        db=conf[active_environment]["SQLDB"],
    )
    return cnx

@Decorators.determine_environment
def get_active_environment():
    '''
    Reads configuration file from ./settings/configuration.ini
    Returns ["DEFAULT"]["ActiveEnvironment"]
    '''
    conf = configparser.ConfigParser()
    conf.read("./settings/configuration.ini")
    return conf["DEFAULT"]["ActiveEnvironment"]


@Decorators.determine_environment
def raw_connection():
    con_str = construct_con_str()
    engine = create_engine(con_str)
    return engine


def construct_con_str():
    active_environment = get_active_environment()
    conf = configparser.ConfigParser()
    conf.read("./settings/sensitive.ini")
    return (
        "mysql+pymysql://"
        f"{conf[active_environment]['SQLUsername']}"
        f":{conf[active_environment]['SQLPassword']}"
        f"@{conf[active_environment]['SQLHost']}"
        f":{int(conf[active_environment]['SQLPort'])}"
        f"/{conf[active_environment]['SQLDB']}"
    )
    
