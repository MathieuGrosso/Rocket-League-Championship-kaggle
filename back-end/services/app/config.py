import os
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

engine = create_engine("mysql+mysqlconnector://root:Jim95Bosto@127.0.0.1/arolya.db")
# print(database_exists(engine.url))
if not database_exists(engine.url):
    create_database(engine.url)

print(database_exists(engine.url))


class Config(object):
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:Jim95Bosto@127.0.0.1/arolya.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    