from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from hashlib import md5
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    created_on = db.Column(db.DateTime, index=True, default=datetime.now)
    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(digest, size)

    def __repr__(self):
        return '<User {}>'.format(self.username)   



class Investment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stock = db.Column(db.String)
    price = db.Column(db.Integer, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    last_action = db.Column(db.String,default='buy')
    transaction_on = db.Column(db.DateTime, index=True, default=datetime.now)
    
    def __repr__(self):
        return self.message
