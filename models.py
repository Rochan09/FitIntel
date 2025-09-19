from flask_sqlalchemy import SQLAlchemy

db=SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable= False)
    email = db.Column(db.String(100), unique=True, nullable= False)
    password = db.Column(db.String(60), nullable= False)
