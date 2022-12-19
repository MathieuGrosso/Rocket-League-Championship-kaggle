import os
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import redis 

from dotenv import load_dotenv
load_dotenv()


engine = create_engine("mysql+mysqlconnector://root:Jim95Bosto@127.0.0.1/arolya.db")
# print(database_exists(engine.url))
if not database_exists(engine.url):
    create_database(engine.url)




class Config(object):
    SECRET_KEY = os.environ["SECRET_KEY"]
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:Jim95Bosto@127.0.0.1/arolya.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_TYPE='filesystem'
    SESSION_PERMANENT=False
    SESSION_USE_SIGNER=True
    SESSION_REDIS = redis.from_url('redis://127.0.0.1:6379')

    