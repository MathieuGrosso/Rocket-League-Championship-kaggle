from flask import Flask, jsonify, request, abort, session
import os
from flask_session import Session
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from flask_cors import CORS
from app import app, db, bcrypt, server_session
from models import User


@app.route("/@me", methods=['GET'])
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"message": "User not logged in"}), 401

    user = User.query.filter_by(id=user_id).first()
    if user is None:
        return jsonify({"message": "User not found"}), 404

    return jsonify({"user": user.email,
                    "username": user.username,
                    "id": user.id,
                    "created_at": user.created_at})


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

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Wrong password"}), 401
    session["user_id"] = user.id

    return jsonify({"message": f"User {session['user_id']} Logged in"})
