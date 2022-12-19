
#!/usr/bin/env python
from flask import Flask, jsonify, request, abort
import os
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func


app = Flask(__name__)

bcrypt = Bcrypt(app)

app.config.from_object("config.Config")

db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(128), unique=True, nullable=False)
    active = db.Column(db.Boolean(), default=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())
    username = db.Column(db.String(128), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __init__(self, email, password, username):
        self.email = email
        self.username = username
        self.password = password

    def __repr__(self):
        return f'<User {self.email}>'


@app.route('/', methods=['GET'])
def get_hello():
    return jsonify({"hello": "world"})


@app.route("/register", methods=['POST'])
def register_user():
    data = request.get_json()

    user_exists = User.query.filter_by(email=data["email"]).first()
    if user_exists:
        return jsonify({"message": "User already exists"})

    user_password = data["password"]
    hashed_password = bcrypt.generate_password_hash(
        user_password).decode('utf-8')
    user = User(data["email"],  hashed_password, data["username"])

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User created"})


@app.route("/login", methods=['POST'])
def login_user():
    data = request.get_json()
    password = data["password"]
    email = data["email"]
    username = data["username"]

    user = User.query.filter_by(email=email).first()

    if user is None:
        return jsonify({"message": "User does not exist"}), 401

    if bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "User logged in"}), 200
    else:
        return jsonify({"message": "Wrong password"}), 401


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
