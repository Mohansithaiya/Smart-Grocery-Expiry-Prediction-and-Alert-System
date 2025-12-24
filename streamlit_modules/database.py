# Database utilities (MongoDB and SQLite)

import sqlite3
from pymongo import MongoClient
import os


def init_mongodb():
    uri = os.getenv('MONGODB_URI', 'mongodb://127.0.0.1:27017/')
    client = MongoClient(uri)
    return client['grocery_app']


def init_db(user_id):
    db_path = f'grocery_{user_id}.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            manufacture_date TEXT,
            expiry_date TEXT
        )
    ''')
    conn.commit()
    conn.close()
    return db_path
