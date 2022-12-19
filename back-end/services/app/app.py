
#!/usr/bin/env python
from flask import Flask, jsonify, request, abort, session
import os
from flask_session import Session
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from flask_cors import CORS, cross_origin

app = Flask(__name__)
bcrypt = Bcrypt(app)
CORS(app, supports_credentials=True)
app.config.from_object("config.Config")
db = SQLAlchemy(app)

server_session = Session(app)


@app.route('/', methods=['GET'])
def get_hello():
    return jsonify({"hello": "world"})
