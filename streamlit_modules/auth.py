# Authentication logic

import streamlit as st
from streamlit_cookies_manager import CookieManager
import bcrypt
import uuid

cookies = CookieManager()


def init_auth():
    return cookies.ready()


def create_user(db, email, password, name):
    users = db['users']
    if users.find_one({'email': email}):
        return False
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        '_id': str(uuid.uuid4()),
        'email': email,
        'password': hashed,
        'name': name
    })
    return True


def verify_user(db, email, password):
    user = db['users'].find_one({'email': email})
    if user and bcrypt.checkpw(password.encode(), user['password']):
        return True, user
    return False, None
