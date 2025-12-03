import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_bytes
import tempfile
import re
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import schedule
import time
import threading
import json
from pymongo import MongoClient
import bcrypt
import uuid
from streamlit_cookies_manager import CookieManager
import requests
from io import StringIO
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Test openpyxl availability
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Load environment variables
load_dotenv()

# Set page configuration with better theme
st.set_page_config(
    page_title="Grocery Expiry Alert System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #F8F9FA;
        border-left: 5px solid #2E86AB;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f46b45 0%, #eea849 100%);
        color: white;
    }
    .danger-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #A23B72 0%, #2E86AB 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MongoDB connection
def init_mongodb():
    try:
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://127.0.0.1:27017/expiry-alert')
        client = MongoClient(mongodb_uri)
        db = client['grocery_app']
        return db
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

# Initialize cookies
cookies = CookieManager()

# Authentication functions
def init_auth():
    if not cookies.ready():
        st.error("Cookies not ready. Please refresh the page.")
        return False
    return True

def create_user(db, email, password, name, alert_days_before=1):
    users_collection = db['users']
    
    # Check if user already exists
    if users_collection.find_one({'email': email}):
        return False, "User already exists"
    
    # Hash password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Create user document
    user = {
        '_id': str(uuid.uuid4()),
        'email': email,
        'password': hashed_password,
        'name': name,
        'alert_days_before': alert_days_before,
        'created_at': datetime.now(),
        'email_verified': False
    }
    
    # Insert user
    users_collection.insert_one(user)
    
    # Send welcome email
    send_welcome_email(email, name)
    
    return True, "User created successfully"

def verify_user(db, email, password):
    users_collection = db['users']
    user = users_collection.find_one({'email': email})
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True, user
    return False, None

def update_user_alert_preference(db, user_id, alert_days_before):
    users_collection = db['users']
    users_collection.update_one(
        {'_id': user_id},
        {'$set': {'alert_days_before': alert_days_before}}
    )

def send_welcome_email(email, name):
    subject = "Welcome to Grocery Expiry Alert System"
    body = f"""
    <html>
    <body>
        <h2>Welcome, {name}!</h2>
        <p>Thank you for registering with the Grocery Expiry Alert System.</p>
        <p>You will now receive alerts about your grocery items before they expire.</p>
        <br>
        <p>Best regards,<br>Grocery Expiry Alert System Team</p>
    </body>
    </html>
    """
    
    send_email(email, subject, body)

# Initialize database
def init_db(user_id):
    db_path = f'grocery_{user_id}.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS items
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 category TEXT NOT NULL,
                 manufacture_date DATE NOT NULL,
                 expiry_date DATE NOT NULL,
                 storage_type TEXT DEFAULT 'pantry',
                 alert_days_before INTEGER DEFAULT 1,
                 notified INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()
    return db_path

# Branded Product Database with Shelf Life Information
class BrandedProductDatabase:
    def __init__(self):
        self.products = {
            # Maggi Products
            'maggi noodles': {'category': 'Ready-to-Eat', 'shelf_life_days': 80, 'brand': 'Maggi', 'storage': 'pantry'},
            'maggi 2-minute noodles': {'category': 'Ready-to-Eat', 'shelf_life_days': 80, 'brand': 'Maggi', 'storage': 'pantry'},
            'maggi masala': {'category': 'Masala', 'shelf_life_days': 365, 'brand': 'Maggi', 'storage': 'pantry'},
            'maggi tomato ketchup': {'category': 'Condiments', 'shelf_life_days': 365, 'brand': 'Maggi', 'storage': 'pantry'},
            
            # Cadbury Products
            'cadbury dairy milk': {'category': 'Chocolates', 'shelf_life_days': 180, 'brand': 'Cadbury', 'storage': 'pantry'},
            'cadbury 5-star': {'category': 'Chocolates', 'shelf_life_days': 180, 'brand': 'Cadbury', 'storage': 'pantry'},
            'cadbury perk': {'category': 'Chocolates', 'shelf_life_days': 180, 'brand': 'Cadbury', 'storage': 'pantry'},
            'cadbury gems': {'category': 'Chocolates', 'shelf_life_days': 180, 'brand': 'Cadbury', 'storage': 'pantry'},
            
            # Nestle Products
            'nestle milkmaid': {'category': 'Dairy', 'shelf_life_days': 730, 'brand': 'Nestle', 'storage': 'pantry'},
            'nestle kitkat': {'category': 'Chocolates', 'shelf_life_days': 365, 'brand': 'Nestle', 'storage': 'pantry'},
            'nestle nescafe': {'category': 'Beverages', 'shelf_life_days': 730, 'brand': 'Nestle', 'storage': 'pantry'},
            
            # Amul Products
            'amul butter': {'category': 'Dairy', 'shelf_life_days': 90, 'brand': 'Amul', 'storage': 'refrigerator'},
            'amul cheese': {'category': 'Dairy', 'shelf_life_days': 60, 'brand': 'Amul', 'storage': 'refrigerator'},
            'amul milk': {'category': 'Dairy', 'shelf_life_days': 3, 'brand': 'Amul', 'storage': 'refrigerator'},
            'amul yogurt': {'category': 'Dairy', 'shelf_life_days': 7, 'brand': 'Amul', 'storage': 'refrigerator'},
            
            # Patanjali Products
            'patanjali honey': {'category': 'Sweeteners', 'shelf_life_days': 730, 'brand': 'Patanjali', 'storage': 'pantry'},
            'patanjali ghee': {'category': 'Cooking Oil', 'shelf_life_days': 180, 'brand': 'Patanjali', 'storage': 'pantry'},
            'patanjali mustard oil': {'category': 'Cooking Oil', 'shelf_life_days': 365, 'brand': 'Patanjali', 'storage': 'pantry'},
            
            # Dabur Products
            'dabur honey': {'category': 'Sweeteners', 'shelf_life_days': 730, 'brand': 'Dabur', 'storage': 'pantry'},
            'dabur chyawanprash': {'category': 'Health Supplements', 'shelf_life_days': 1095, 'brand': 'Dabur', 'storage': 'pantry'},
            'dabur amla juice': {'category': 'Beverages', 'shelf_life_days': 365, 'brand': 'Dabur', 'storage': 'refrigerator'},
            
            # Britannia Products
            'britannia good day': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Britannia', 'storage': 'pantry'},
            'britannia marie gold': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Britannia', 'storage': 'pantry'},
            'britannia tiger': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Britannia', 'storage': 'pantry'},
            
            # Parle Products
            'parle-g': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Parle', 'storage': 'pantry'},
            'parle monaco': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Parle', 'storage': 'pantry'},
            'parle krackjack': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Parle', 'storage': 'pantry'},
            
            # Tata Products
            'tata salt': {'category': 'Condiments', 'shelf_life_days': 1095, 'brand': 'Tata', 'storage': 'pantry'},
            'tata tea': {'category': 'Beverages', 'shelf_life_days': 730, 'brand': 'Tata', 'storage': 'pantry'},
            'tata namak': {'category': 'Condiments', 'shelf_life_days': 1095, 'brand': 'Tata', 'storage': 'pantry'},
            
            # Fortune Products
            'fortune sunflower oil': {'category': 'Cooking Oil', 'shelf_life_days': 365, 'brand': 'Fortune', 'storage': 'pantry'},
            'fortune rice bran oil': {'category': 'Cooking Oil', 'shelf_life_days': 365, 'brand': 'Fortune', 'storage': 'pantry'},
            'fortune mustard oil': {'category': 'Cooking Oil', 'shelf_life_days': 365, 'brand': 'Fortune', 'storage': 'pantry'},
            
            # Knorr Products
            'knorr soup': {'category': 'Ready-to-Eat', 'shelf_life_days': 730, 'brand': 'Knorr', 'storage': 'pantry'},
            'knorr tomato soup': {'category': 'Ready-to-Eat', 'shelf_life_days': 730, 'brand': 'Knorr', 'storage': 'pantry'},
            'knorr chicken soup': {'category': 'Ready-to-Eat', 'shelf_life_days': 730, 'brand': 'Knorr', 'storage': 'pantry'},
            
            # ITC Products
            'sunfeast biscuits': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'ITC', 'storage': 'pantry'},
            'sunfeast dark fantasy': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'ITC', 'storage': 'pantry'},
            'bingo chips': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'ITC', 'storage': 'pantry'},
            
            # Haldiram Products
            'haldiram namkeen': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Haldiram', 'storage': 'pantry'},
            'haldiram bhujia': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Haldiram', 'storage': 'pantry'},
            'haldiram sweets': {'category': 'Sweets', 'shelf_life_days': 30, 'brand': 'Haldiram', 'storage': 'refrigerator'},
            
            # Coca Cola Products
            'coca cola': {'category': 'Beverages', 'shelf_life_days': 180, 'brand': 'Coca Cola', 'storage': 'pantry'},
            'pepsi': {'category': 'Beverages', 'shelf_life_days': 180, 'brand': 'Pepsi', 'storage': 'pantry'},
            'sprite': {'category': 'Beverages', 'shelf_life_days': 180, 'brand': 'Coca Cola', 'storage': 'pantry'},
            
            # Lays Products
            'lays chips': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Lays', 'storage': 'pantry'},
            'lays classic': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Lays', 'storage': 'pantry'},
            'lays magic masala': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Lays', 'storage': 'pantry'},
            
            # Kurkure Products
            'kurkure': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Kurkure', 'storage': 'pantry'},
            'kurkure masala munch': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Kurkure', 'storage': 'pantry'},
            
            # More branded products
            'horlicks': {'category': 'Health Supplements', 'shelf_life_days': 730, 'brand': 'Horlicks', 'storage': 'pantry'},
            'bournvita': {'category': 'Health Supplements', 'shelf_life_days': 730, 'brand': 'Bournvita', 'storage': 'pantry'},
            'complan': {'category': 'Health Supplements', 'shelf_life_days': 730, 'brand': 'Complan', 'storage': 'pantry'},
            'boost': {'category': 'Health Supplements', 'shelf_life_days': 730, 'brand': 'Boost', 'storage': 'pantry'},
            
            # Biscuit brands
            'oreo': {'category': 'Biscuits', 'shelf_life_days': 365, 'brand': 'Oreo', 'storage': 'pantry'},
            'hide and seek': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Parle', 'storage': 'pantry'},
            'monaco salt': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Parle', 'storage': 'pantry'},
            
            # Cooking essentials
            'tata sambar powder': {'category': 'Masala', 'shelf_life_days': 365, 'brand': 'Tata', 'storage': 'pantry'},
            'everest garam masala': {'category': 'Masala', 'shelf_life_days': 365, 'brand': 'Everest', 'storage': 'pantry'},
            'everest pav bhaji masala': {'category': 'Masala', 'shelf_life_days': 365, 'brand': 'Everest', 'storage': 'pantry'},
            'mdh garam masala': {'category': 'Masala', 'shelf_life_days': 365, 'brand': 'MDH', 'storage': 'pantry'},
            
            # Frozen foods
            'amul ice cream': {'category': 'Frozen Foods', 'shelf_life_days': 180, 'brand': 'Amul', 'storage': 'freezer'},
            'kwality walls': {'category': 'Frozen Foods', 'shelf_life_days': 180, 'brand': 'Kwality Walls', 'storage': 'freezer'},
            
            # Instant mixes
            'mtr idli mix': {'category': 'Instant Mix', 'shelf_life_days': 180, 'brand': 'MTR', 'storage': 'pantry'},
            'mtr dosa mix': {'category': 'Instant Mix', 'shelf_life_days': 180, 'brand': 'MTR', 'storage': 'pantry'},
            'mtr upma mix': {'category': 'Instant Mix', 'shelf_life_days': 180, 'brand': 'MTR', 'storage': 'pantry'},
            
            # Ready to eat
            'mtr ready to eat': {'category': 'Ready-to-Eat', 'shelf_life_days': 365, 'brand': 'MTR', 'storage': 'pantry'},
            'kohinoor ready to eat': {'category': 'Ready-to-Eat', 'shelf_life_days': 365, 'brand': 'Kohinoor', 'storage': 'pantry'},
            
            # Noodles
            'yippee noodles': {'category': 'Noodles', 'shelf_life_days': 80, 'brand': 'Yippee', 'storage': 'pantry'},
            'top ramen': {'category': 'Noodles', 'shelf_life_days': 80, 'brand': 'Top Ramen', 'storage': 'pantry'},
            'wai wai noodles': {'category': 'Noodles', 'shelf_life_days': 80, 'brand': 'Wai Wai', 'storage': 'pantry'},
            
            # Beverages
            'tropicana juice': {'category': 'Beverages', 'shelf_life_days': 30, 'brand': 'Tropicana', 'storage': 'refrigerator'},
            'real juice': {'category': 'Beverages', 'shelf_life_days': 30, 'brand': 'Real', 'storage': 'refrigerator'},
            'maaza': {'category': 'Beverages', 'shelf_life_days': 180, 'brand': 'Maaza', 'storage': 'pantry'},
            'frooti': {'category': 'Beverages', 'shelf_life_days': 180, 'brand': 'Frooti', 'storage': 'pantry'},
            
            # Dairy products
            'mother dairy milk': {'category': 'Dairy', 'shelf_life_days': 3, 'brand': 'Mother Dairy', 'storage': 'refrigerator'},
            'heritage milk': {'category': 'Dairy', 'shelf_life_days': 3, 'brand': 'Heritage', 'storage': 'refrigerator'},
            'aavin milk': {'category': 'Dairy', 'shelf_life_days': 3, 'brand': 'Aavin', 'storage': 'refrigerator'},
        }
    
    def get_product_info(self, product_name):
        """Get product information by name (case-insensitive)"""
        product_lower = product_name.lower().strip()
        
        # Direct match
        if product_lower in self.products:
            return self.products[product_lower]
        
        # Partial match
        for key, value in self.products.items():
            if product_lower in key or key in product_lower:
                return value
        
        return None
    
    def is_product_available(self, product_name):
        """Check if product exists in database"""
        return self.get_product_info(product_name) is not None
    
    def get_similar_products(self, product_name, limit=5):
        """Get similar products for suggestions"""
        product_lower = product_name.lower().strip()
        similar = []
        
        for key in self.products.keys():
            if any(word in key for word in product_lower.split()):
                similar.append(key)
        
        return similar[:limit]

# Initialize branded product database
branded_db = BrandedProductDatabase()

# Enhanced search helper functions
def get_autocomplete_suggestions(query, limit=10):
    """Get autocomplete suggestions for product search"""
    if not query or len(query) < 1:
        return []
    
    query_lower = query.lower().strip()
    suggestions = []
    
    # Exact matches first
    for product_key in branded_db.products.keys():
        if product_key.startswith(query_lower):
            product_info = branded_db.products[product_key]
            suggestions.append({
                'product': product_key,
                'brand': product_info['brand'],
                'category': product_info['category'],
                'shelf_life': product_info['shelf_life_days'],
                'match_type': 'exact_start'
            })
    
    # Partial matches
    if len(suggestions) < limit:
        for product_key in branded_db.products.keys():
            if query_lower in product_key and product_key not in [s['product'] for s in suggestions]:
                product_info = branded_db.products[product_key]
                suggestions.append({
                    'product': product_key,
                    'brand': product_info['brand'],
                    'category': product_info['category'],
                    'shelf_life': product_info['shelf_life_days'],
                    'match_type': 'partial'
                })
    
    # Brand matches
    if len(suggestions) < limit:
        for product_key, product_info in branded_db.products.items():
            if (query_lower in product_info['brand'].lower() and 
                product_key not in [s['product'] for s in suggestions]):
                suggestions.append({
                    'product': product_key,
                    'brand': product_info['brand'],
                    'category': product_info['category'],
                    'shelf_life': product_info['shelf_life_days'],
                    'match_type': 'brand'
                })
    
    # Sort by match type and alphabetical
    suggestions.sort(key=lambda x: (x['match_type'] == 'exact_start', x['product']))
    return suggestions[:limit]

def display_product_suggestions(suggestions, input_key="item_name_input"):
    """Display product suggestions with clickable buttons"""
    if not suggestions:
        return
    
    st.markdown("**💡 Product Suggestions:**")
    
    for i, suggestion in enumerate(suggestions):
        col_a, col_b, col_c = st.columns([3, 1, 1])
        
        with col_a:
            # Display product info with match type indicator
            match_icon = "🎯" if suggestion['match_type'] == 'exact_start' else "📦" if suggestion['match_type'] == 'partial' else "🏷️"
            st.markdown(f"{match_icon} **{suggestion['product'].title()}** ({suggestion['brand']}) - {suggestion['shelf_life']} days")
        
        with col_b:
            # Click to select button - use a different session state key
            button_key = f"select_{input_key}_{i}"
            if st.button("Select", key=button_key, help=f"Click to auto-fill '{suggestion['product'].title()}'"):
                # Store the selected product in a different session state key
                st.session_state[f"selected_{input_key}"] = suggestion['product'].title()
                st.rerun()
        
        with col_c:
            # Show category
            st.text(suggestion['category'])

# Enhanced ML Model with real-world data simulation
class EnhancedExpiryPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = ['category', 'storage_type', 'manufacture_month', 'temperature', 'humidity']
        
    def create_realistic_dataset(self):
        """Create a more realistic dataset based on food safety guidelines"""
        categories = {
            'Dairy': ['milk', 'yogurt', 'cheese', 'butter', 'cream'],
            'Meat': ['chicken', 'beef', 'pork', 'fish', 'lamb'],
            'Fruits': ['apples', 'bananas', 'oranges', 'berries', 'grapes'],
            'Vegetables': ['lettuce', 'tomatoes', 'carrots', 'broccoli', 'spinach'],
            'Bakery': ['bread', 'cakes', 'pastries', 'buns'],
            'Grains': ['rice', 'pasta', 'cereal', 'flour'],
            'Beverages': ['juice', 'soda', 'water', 'tea']
        }
        
        data = []
        np.random.seed(42)
        
        for category, items in categories.items():
            for item in items:
                for month in range(1, 13):
                    for storage in ['pantry', 'refrigerator', 'freezer']:
                        # Base shelf life varies by category and storage
                        base_shelf_life = self.get_base_shelf_life(category, storage)
                        
                        # Seasonal effects (summer vs winter)
                        seasonal_factor = 1.0
                        if month in [6, 7, 8]:  # Summer
                            if storage == 'pantry':
                                seasonal_factor = 0.7  # Reduced shelf life in summer
                        
                        # Temperature and humidity simulation
                        if storage == 'pantry':
                            temp = np.random.normal(25, 5)  # Room temperature
                            humidity = np.random.normal(60, 10)
                        elif storage == 'refrigerator':
                            temp = np.random.normal(4, 1)   # Refrigerator temperature
                            humidity = np.random.normal(80, 5)
                        else:  # freezer
                            temp = np.random.normal(-18, 2) # Freezer temperature
                            humidity = np.random.normal(20, 5)
                        
                        # Calculate shelf life with variations
                        shelf_life = base_shelf_life * seasonal_factor
                        shelf_life += np.random.normal(0, shelf_life * 0.1)  # Random variation
                        
                        # Ensure minimum shelf life
                        shelf_life = max(1, shelf_life)
                        
                        data.append({
                            'item_name': item,
                            'category': category,
                            'manufacture_month': month,
                            'storage_type': storage,
                            'temperature': temp,
                            'humidity': humidity,
                            'shelf_life_days': int(shelf_life)
                        })
        
        df = pd.DataFrame(data)
        df.to_csv('enhanced_shelf_life_data.csv', index=False)
        return df
    
    def get_base_shelf_life(self, category, storage):
        """Get base shelf life based on category and storage type"""
        base_life = {
            'Dairy': {'pantry': 2, 'refrigerator': 14, 'freezer': 90},
            'Meat': {'pantry': 1, 'refrigerator': 5, 'freezer': 180},
            'Fruits': {'pantry': 7, 'refrigerator': 14, 'freezer': 60},
            'Vegetables': {'pantry': 5, 'refrigerator': 10, 'freezer': 90},
            'Bakery': {'pantry': 5, 'refrigerator': 7, 'freezer': 30},
            'Grains': {'pantry': 365, 'refrigerator': 365, 'freezer': 365},
            'Beverages': {'pantry': 180, 'refrigerator': 180, 'freezer': 180}
        }
        return base_life.get(category, {}).get(storage, 7)
    
    def train_enhanced_model(self, data=None):
        try:
            if data is None:
                if os.path.exists('enhanced_shelf_life_data.csv'):
                    df = pd.read_csv('enhanced_shelf_life_data.csv')
                else:
                    df = self.create_realistic_dataset()
            else:
                df = data
            
            # Feature engineering
            df['manufacture_month'] = df['manufacture_month'].astype(int)
            df['storage_type'] = df['storage_type'].fillna('pantry')
            
            # Encode categorical variables
            for feature in ['category', 'storage_type']:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature])
                self.label_encoders[feature] = le
            
            # Prepare features and target
            X = df[['category', 'storage_type', 'manufacture_month', 'temperature', 'humidity']]
            y = df['shelf_life_days']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train multiple models and choose the best
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            best_score = 0
            best_model = None
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            self.model = best_model
            
            # Save model with metrics
            model_info = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'r2_score': best_score,
                'mae': mean_absolute_error(y_test, y_pred),
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('enhanced_expiry_model.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            return True, f"Model trained successfully! R² Score: {best_score:.3f}, MAE: {mean_absolute_error(y_test, y_pred):.2f} days"
            
        except Exception as e:
            return False, f"Error training model: {e}"
    
    def load_model(self):
        try:
            if os.path.exists('enhanced_expiry_model.pkl'):
                with open('enhanced_expiry_model.pkl', 'rb') as f:
                    model_info = pickle.load(f)
                    self.model = model_info['model']
                    self.label_encoders = model_info['label_encoders']
                    self.scaler = model_info['scaler']
                    self.feature_names = model_info['feature_names']
                return True
            return False
        except:
            return False
    
    def predict_expiry(self, item_name, category, manufacture_date, storage_type='pantry', temperature=None, humidity=None):
        if not self.model:
            if not self.load_model():
                return self.fallback_prediction(item_name, manufacture_date, storage_type)
        
        try:
            # Get default temperature/humidity if not provided
            if temperature is None:
                temperature = self.get_default_temperature(storage_type)
            if humidity is None:
                humidity = self.get_default_humidity(storage_type)
            
            # Prepare features
            manufacture_month = manufacture_date.month
            
            # Encode categorical variables
            category_encoded = self.label_encoders['category'].transform([category])[0]
            storage_encoded = self.label_encoders['storage_type'].transform([storage_type])[0]
            
            # Create feature array
            features = np.array([[category_encoded, storage_encoded, manufacture_month, temperature, humidity]])
            features_scaled = self.scaler.transform(features)
            
            # Predict shelf life
            shelf_life_days = self.model.predict(features_scaled)[0]
            
            return manufacture_date + timedelta(days=max(1, int(shelf_life_days)))
            
        except Exception as e:
            st.warning(f"Using fallback prediction due to error: {e}")
            return self.fallback_prediction(item_name, manufacture_date, storage_type)
    
    def get_default_temperature(self, storage_type):
        temps = {'pantry': 25, 'refrigerator': 4, 'freezer': -18}
        return temps.get(storage_type, 25)
    
    def get_default_humidity(self, storage_type):
        humidities = {'pantry': 60, 'refrigerator': 80, 'freezer': 20}
        return humidities.get(storage_type, 60)
    
    def fallback_prediction(self, item_name, manufacture_date, storage_type):
        # Enhanced fallback with more items
        default_shelf_life = {
            'milk': {'pantry': 2, 'refrigerator': 7, 'freezer': 90},
            'bread': {'pantry': 5, 'refrigerator': 7, 'freezer': 30},
            'eggs': {'pantry': 3, 'refrigerator': 21, 'freezer': 365},
            'yogurt': {'pantry': 2, 'refrigerator': 14, 'freezer': 60},
            'cheese': {'pantry': 3, 'refrigerator': 30, 'freezer': 180},
            'chicken': {'pantry': 1, 'refrigerator': 3, 'freezer': 270},
            'beef': {'pantry': 1, 'refrigerator': 5, 'freezer': 365},
            'fish': {'pantry': 1, 'refrigerator': 2, 'freezer': 180},
            'apples': {'pantry': 30, 'refrigerator': 60, 'freezer': 240},
            'bananas': {'pantry': 7, 'refrigerator': 10, 'freezer': 60},
            'oranges': {'pantry': 21, 'refrigerator': 30, 'freezer': 120},
            'tomatoes': {'pantry': 7, 'refrigerator': 14, 'freezer': 60},
            'lettuce': {'pantry': 3, 'refrigerator': 7, 'freezer': 30},
            'carrots': {'pantry': 21, 'refrigerator': 30, 'freezer': 270},
            'potatoes': {'pantry': 90, 'refrigerator': 120, 'freezer': 365},
            'onions': {'pantry': 60, 'refrigerator': 90, 'freezer': 240},
            'broccoli': {'pantry': 3, 'refrigerator': 7, 'freezer': 365},
            'spinach': {'pantry': 3, 'refrigerator': 5, 'freezer': 240}
        }
        
        item_lower = item_name.lower()
        for item, storage_life in default_shelf_life.items():
            if item in item_lower:
                days = storage_life.get(storage_type, storage_life.get('pantry', 7))
                return manufacture_date + timedelta(days=days)
        
        return manufacture_date + timedelta(days=7)

# Initialize the enhanced predictor
enhanced_predictor = EnhancedExpiryPredictor()

# OCR functions (same as before)
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    try:
        images = convert_from_bytes(pdf_file.read())
        text = ""
        for image in images:
            text += extract_text_from_image(image) + "\n"
        return text
    except Exception as e:
        st.error(f"PDF Processing Error: {e}")
        return ""

def parse_receipt_text(text):
    """Enhanced receipt parsing with better date and product recognition"""
    
    # Enhanced date patterns
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{2,4}',
        r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}',
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{2,4}'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    manufacture_date = None
    for date_str in dates:
        try:
            # Try different date formats
            for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y']:
                try:
                    manufacture_date = datetime.strptime(date_str, fmt)
                    break
                except:
                    continue
            if manufacture_date:
                break
        except:
            continue
    
    if not manufacture_date:
        manufacture_date = datetime.now()
    
    # Enhanced product recognition using branded database
    items = []
    lines = text.split('\n')
    
    for line in lines:
        line_clean = line.strip()
        if len(line_clean) < 3 or len(line_clean) > 100:  # Skip very short or very long lines
            continue
            
        line_lower = line_clean.lower()
        
        # Check against branded product database
        for product_key in branded_db.products.keys():
            # Check if product name appears in the line
            if product_key in line_lower or any(word in line_lower for word in product_key.split()):
                product_info = branded_db.products[product_key]
                
                # Avoid duplicates
                if not any(item['name'].lower() == product_key for item in items):
                    items.append({
                        'name': product_key.title(),
                        'category': product_info['category'],
                        'brand': product_info['brand']
                    })
        
        # Also check for common generic products
        generic_products = {
            'milk': 'Dairy',
            'bread': 'Bakery',
            'eggs': 'Dairy',
            'rice': 'Grains',
            'oil': 'Cooking Oil',
            'salt': 'Condiments',
            'sugar': 'Sweeteners',
            'flour': 'Grains',
            'onion': 'Vegetables',
            'tomato': 'Vegetables',
            'potato': 'Vegetables',
            'apple': 'Fruits',
            'banana': 'Fruits',
            'orange': 'Fruits'
        }
        
        for generic_item, category in generic_products.items():
            if generic_item in line_lower and len(line_clean) < 50:
                # Avoid duplicates
                if not any(item['name'].lower() == generic_item for item in items):
                    items.append({
                        'name': generic_item.capitalize(),
                        'category': category,
                        'brand': 'Generic'
                    })
    
    # Remove duplicates and limit to reasonable number
    unique_items = []
    seen_names = set()
    for item in items:
        if item['name'].lower() not in seen_names:
            unique_items.append(item)
            seen_names.add(item['name'].lower())
            if len(unique_items) >= 20:  # Limit to 20 items max
                break
    
    return manufacture_date, unique_items

# Email functions
def send_email(to_email, subject, body):
    try:
        from_email = os.getenv('EMAIL_FROM')
        password = os.getenv('EMAIL_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 465))
        
        if not from_email or not password:
            st.error("Email credentials not configured. Please check your .env file.")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

def check_and_notify(user_id, user_email, alert_days_before=1):
    db_path = f'grocery_{user_id}.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    alert_date = datetime.now() + timedelta(days=alert_days_before)
    alert_date_str = alert_date.strftime('%Y-%m-%d')
    
    c.execute("SELECT * FROM items WHERE expiry_date = ? AND notified = 0", (alert_date_str,))
    expiring_items = c.fetchall()
    
    if expiring_items:
        item_list = "\n".join([f"- {item[1]} (Expires: {item[4]})" for item in expiring_items])
        email_body = f"""
        <html>
        <body>
            <h2>Grocery Expiry Alert</h2>
            <p>The following items in your inventory will expire in {alert_days_before} day(s):</p>
            <ul>
            {item_list}
            </ul>
            <p>Please consume them soon or consider discarding them.</p>
            <br>
            <p>Best regards,<br>Grocery Expiry Alert System</p>
        </body>
        </html>
        """
        
        if send_email(user_email, f"Grocery Expiry Alert - {alert_days_before} Day(s) Notice", email_body):
            item_ids = [str(item[0]) for item in expiring_items]
            c.execute(f"UPDATE items SET notified = 1 WHERE id IN ({','.join(item_ids)})")
            conn.commit()
            st.success(f"Notification sent for {len(expiring_items)} items!")
        else:
            st.error("Failed to send notification email.")
    
    conn.close()

# Scheduler function
def run_scheduler(db):
    users_collection = db['users']
    users = users_collection.find({})
    
    for user in users:
        user_id = user['_id']
        user_email = user['email']
        alert_days_before = user.get('alert_days_before', 1)
        schedule.every().day.at("09:00").do(check_and_notify, user_id=user_id, user_email=user_email, alert_days_before=alert_days_before)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def start_scheduler(db):
    scheduler_thread = threading.Thread(target=run_scheduler, args=(db,), daemon=True)
    scheduler_thread.start()

# Enhanced UI Components
def create_metric_card(value, label, card_type="normal"):
    color_class = {
        "normal": "metric-card",
        "success": "metric-card success-card",
        "warning": "metric-card warning-card",
        "danger": "metric-card danger-card"
    }.get(card_type, "metric-card")
    
    return f"""
    <div class="{color_class}">
        <h3>{value}</h3>
        <p>{label}</p>
    </div>
    """

def display_dashboard(df, user_alert_days):
    st.markdown("### 📊 Comprehensive Expiry Analytics Dashboard")
    
    if not df.empty:
        try:
            # Convert date columns
            df['expiry_date'] = pd.to_datetime(df['Expiry Date'], errors='coerce')
            df = df.dropna(subset=['expiry_date'])
            
            if df.empty:
                st.info("No items with valid dates to display.")
                return
            
            # Calculate days until expiry
            today = pd.Timestamp.now().normalize()
            df['days_until_expiry'] = (df['expiry_date'] - today).dt.days
            
        except Exception:
            st.error("Error processing dates")
            return
        
        # Calculate comprehensive metrics
        expired_count = len(df[df['days_until_expiry'] < 0])
        expiring_soon = len(df[(df['days_until_expiry'] >= 0) & (df['days_until_expiry'] <= user_alert_days)])
        safe_items = len(df[df['days_until_expiry'] > user_alert_days])
        total_items = len(df)
        
        # Calculate additional metrics
        avg_shelf_life = df['days_until_expiry'].mean()
        items_expiring_today = len(df[df['days_until_expiry'] == 0])
        items_expiring_this_week = len(df[(df['days_until_expiry'] >= 0) & (df['days_until_expiry'] <= 7)])
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔄 Total Items", total_items)
        with col2:
            st.metric("✅ Safe Items", safe_items, delta_color="off")
        with col3:
            st.metric("⚠️ Expiring Soon", expiring_soon, delta_color="off")
        with col4:
            st.metric("❌ Expired", expired_count, delta_color="off")
        
        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Expiring Today", items_expiring_today, delta_color="off")
        with col2:
            st.metric("📆 This Week", items_expiring_this_week, delta_color="off")
        with col3:
            st.metric("📊 Avg Shelf Life", f"{avg_shelf_life:.1f} days", delta_color="off")
        with col4:
            st.metric("🏪 Categories", len(df['Category'].unique()), delta_color="off")
        
        # Enhanced charts section
        st.markdown("### 📈 Visual Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Expiry status pie chart with better colors
            status_data = {
                'Status': ['Safe', 'Expiring Soon', 'Expired'],
                'Count': [safe_items, expiring_soon, expired_count],
            }
            
            fig_pie = px.pie(
                values=status_data['Count'],
                names=status_data['Status'],
                title="📊 Expiry Status Distribution",
                color_discrete_map={'Safe': '#2E8B57', 'Expiring Soon': '#FF8C00', 'Expired': '#DC143C'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Category bar chart
            category_counts = df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            fig_bar = px.bar(
                category_counts,
                x='Count',
                y='Category',
                orientation='h',
                title="📦 Items by Category",
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Expiry timeline chart
        st.markdown("### 📅 Expiry Timeline")
        
        # Create timeline data
        timeline_data = []
        for i in range(30):  # Next 30 days
            date = today + timedelta(days=i)
            count = len(df[df['days_until_expiry'] == i])
            if count > 0:
                timeline_data.append({'Date': date, 'Items': count, 'Day': i})
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            fig_timeline = px.bar(
                timeline_df,
                x='Date',
                y='Items',
                title="📈 Items Expiring in Next 30 Days",
                color='Items',
                color_continuous_scale='Reds'
            )
            fig_timeline.update_layout(xaxis_title="Date", yaxis_title="Number of Items")
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No items expiring in the next 30 days.")
        
        # Storage analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Storage type analysis
            storage_counts = df['Storage Type'].value_counts().reset_index()
            storage_counts.columns = ['Storage Type', 'Count']
            
            fig_storage = px.pie(
                storage_counts,
                values='Count',
                names='Storage Type',
                title="🏠 Storage Type Distribution",
                color_discrete_map={'pantry': '#8B4513', 'refrigerator': '#87CEEB', 'freezer': '#4682B4'}
            )
            st.plotly_chart(fig_storage, use_container_width=True)
        
        with col2:
            # Purchase frequency analysis (based on manufacture date)
            df['manufacture_date'] = pd.to_datetime(df['Manufacture Date'], errors='coerce')
            df['manufacture_month'] = df['manufacture_date'].dt.to_period('M')
            monthly_counts = df['manufacture_month'].value_counts().sort_index()
            
            if len(monthly_counts) > 1:
                fig_frequency = px.bar(
                    x=monthly_counts.index.astype(str),
                    y=monthly_counts.values,
                    title="📈 Purchase Frequency (by Manufacture Month)",
                    labels={'x': 'Month', 'y': 'Number of Items'}
                )
                st.plotly_chart(fig_frequency, use_container_width=True)
            else:
                st.info("Not enough data for frequency analysis.")
        
        # Smart recommendations
        st.markdown("### 💡 Smart Recommendations")
        
        # Items expiring soon recommendations
        expiring_items = df[(df['days_until_expiry'] >= 0) & (df['days_until_expiry'] <= user_alert_days)]
        if not expiring_items.empty:
            st.warning("⚠️ **Items Expiring Soon - Consider consuming these first:**")
            for _, item in expiring_items.head(5).iterrows():
                days_left = int(item['days_until_expiry'])
                st.write(f"• **{item['Name']}** - {days_left} day(s) left ({item['Category']})")
        
        # Storage optimization recommendations
        st.info("🏠 **Storage Optimization Tips:**")
        if 'refrigerator' in df['Storage Type'].values:
            st.write("• Keep dairy products and perishables in the refrigerator")
        if 'freezer' in df['Storage Type'].values:
            st.write("• Use freezer for long-term storage of frozen foods")
        if 'pantry' in df['Storage Type'].values:
            st.write("• Store dry goods and non-perishables in pantry")
        
        # Category-specific recommendations
        if 'Dairy' in df['Category'].values:
            st.write("• Dairy products have short shelf life - consume quickly")
        if 'Fruits' in df['Category'].values or 'Vegetables' in df['Category'].values:
            st.write("• Fresh produce should be consumed within a week")
        
    else:
        st.info("No items to display in dashboard.")
# Login/Register page with enhanced UI
def auth_page(db):
    st.markdown('<div class="main-header">🛒 Grocery Expiry Alert System</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🚪 Login", "📝 Register"])
    
    with tab1:
        st.markdown('<div class="sub-header">Login to Your Account</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col1 = st.columns([1, 2, 1])
            with col2:
                email = st.text_input("📧 Email", key="login_email")
                password = st.text_input("🔒 Password", type="password", key="login_password")
                
                if st.button("🚀 Login", use_container_width=True):
                    if email and password:
                        success, user = verify_user(db, email, password)
                        if success:
                            cookies['user_id'] = user['_id']
                            cookies['user_email'] = user['email']
                            cookies['user_name'] = user['name']
                            cookies['alert_days_before'] = str(user.get('alert_days_before', 1))
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid email or password")
                    else:
                        st.warning("Please enter both email and password")
    
    with tab2:
        st.markdown('<div class="sub-header">Create New Account</div>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col1 = st.columns([1, 2, 1])
            with col2:
                name = st.text_input("👤 Full Name", key="reg_name")
                email = st.text_input("📧 Email", key="reg_email")
                password = st.text_input("🔒 Password", type="password", key="reg_password")
                confirm_password = st.text_input("🔒 Confirm Password", type="password", key="reg_confirm_password")
                alert_days_before = st.slider("🔔 Alert Days Before Expiry", min_value=1, max_value=7, value=1, 
                                            help="How many days before expiry would you like to be notified?")
                
                if st.button("📝 Register", use_container_width=True):
                    if name and email and password and confirm_password:
                        if password == confirm_password:
                            success, message = create_user(db, email, password, name, alert_days_before)
                            if success:
                                st.success("Registration successful! Please login.")
                            else:
                                st.error(message)
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.warning("Please fill all fields")

# Main application with enhanced UI
def main_app(db, user_id, user_email, user_name, alert_days_before):
    st.markdown(f'<div class="main-header">🛒 Welcome back, {user_name}!</div>', unsafe_allow_html=True)
    
    # Initialize database for the user
    db_path = init_db(user_id)
    
    # Start scheduler if email is configured
    if os.getenv('EMAIL_FROM') and os.getenv('EMAIL_PASSWORD'):
        start_scheduler(db)
    
    # Sidebar with enhanced UI
    with st.sidebar:
        st.markdown(f"### 👤 User Profile")
        st.markdown(f"**Email:** {user_email}")
        st.markdown(f"**Alert Preference:** {alert_days_before} day(s) before expiry")
        
        # Update alert preference
        new_alert_days = st.slider("Update Alert Days", min_value=1, max_value=7, value=alert_days_before,
                                 help="Change how many days before expiry you want to be notified")
        
        if new_alert_days != alert_days_before:
            if st.button("Update Alert Preference"):
                update_user_alert_preference(db, user_id, new_alert_days)
                cookies['alert_days_before'] = str(new_alert_days)
                st.success("Alert preference updated!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        menu_options = {
            "📦 Manual Entry": "manual_entry",
            "📷 Upload Bill": "upload_bill",
            "👀 View Items": "view_items",
            "📊 Dashboard": "dashboard",
            "🏪 Branded Products": "branded_products",
            "🔔 Test Notification": "test_notification",
            "🤖 ML Model Training": "ml_training",
            "⚙️ Settings": "settings"
        }
        
        choice = st.radio("Go to", list(menu_options.keys()))
        
        if st.button("🚪 Logout", use_container_width=True):
            cookies['user_id'] = ""
            cookies['user_email'] = ""
            cookies['user_name'] = ""
            cookies['alert_days_before'] = ""
            st.rerun()
    
    # Main content area
    if menu_options[choice] == "manual_entry":
        st.markdown('<div class="sub-header">📦 Add Item Manually</div>', unsafe_allow_html=True)
        
        # Product database information
        st.info("💡 **Tip:** Enter branded product names like 'Maggi Noodles', 'Cadbury Dairy Milk', 'Amul Butter' for accurate predictions!")
        
        # Enhanced autocomplete suggestions (outside form)
        # Check if a product was selected from suggestions
        if f"selected_item_name_input" in st.session_state:
            default_value = st.session_state[f"selected_item_name_input"]
            # Clear the selection after using it
            del st.session_state[f"selected_item_name_input"]
        else:
            default_value = ""
        
        item_name_input = st.text_input("Item Name*", 
                                       placeholder="e.g., Maggi Noodles, Cadbury Chocolate, Amul Butter", 
                                       key="item_name_input",
                                       value=default_value)
        
        # Show real-time autocomplete suggestions
        if item_name_input and len(item_name_input) > 0:
            # Get smart autocomplete suggestions
            suggestions = get_autocomplete_suggestions(item_name_input, limit=8)
            
            if suggestions:
                # Display suggestions using the helper function
                display_product_suggestions(suggestions, "item_name_input")
                
                # Show additional info
                if len(suggestions) >= 8:
                    st.info("💡 Showing top 8 matches. Type more letters for specific results.")
            
            else:
                # No matches found - show helpful suggestions
                st.warning("❌ No matching products found.")
                
                # Show popular products by first letter
                first_letter = item_name_input[0].lower()
                popular_products = []
                for product_key in branded_db.products.keys():
                    if product_key.startswith(first_letter):
                        popular_products.append(product_key)
                
                if popular_products:
                    st.info(f"💡 **Products starting with '{first_letter.upper()}':** {len(popular_products)} found")
                    examples = sorted(popular_products)[:5]
                    st.text(f"Examples: {', '.join([ex.title() for ex in examples])}")
                
                # Show category and brand suggestions
                st.info("💡 **Available categories:** Dairy, Ready-to-Eat, Chocolates, Biscuits, Beverages, Snacks, etc.")
                st.info("💡 **Popular brands:** Maggi, Cadbury, Amul, Nestle, Britannia, Parle, Tata, etc.")
                
                # Show quick category buttons
                st.markdown("**🎯 Quick Search by Category:**")
                quick_cats = ["Dairy", "Chocolates", "Biscuits", "Ready-to-Eat"]
                cat_cols = st.columns(len(quick_cats))
                
                for i, category in enumerate(quick_cats):
                    with cat_cols[i]:
                        if st.button(f"📦 {category}", key=f"quick_cat_{category}"):
                            # Find a popular product from this category
                            for product_key, product_info in branded_db.products.items():
                                if product_info['category'] == category:
                                    st.session_state[f"selected_item_name_input"] = product_key.title()
                                    st.rerun()
                                    break
        
        # Form for the remaining inputs
        with st.form("manual_entry_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Use the item name from the input above
                item_name = item_name_input
                manufacture_date = st.date_input("Manufacture Date*", datetime.now())
                storage_type = st.selectbox("Storage Type*", ["Pantry", "Refrigerator", "Freezer"])
            
            with col2:
                custom_alert_days = st.slider("Alert Before Expiry (days)", min_value=1, max_value=14, value=alert_days_before)
            
            if st.form_submit_button("🚀 Predict & Save Item", use_container_width=True):
                if item_name:
                    # Check if product exists in branded database
                    product_info = branded_db.get_product_info(item_name)
                    
                    if product_info:
                        # Use branded product information
                        category = product_info['category']
                        shelf_life_days = product_info['shelf_life_days']
                        brand = product_info['brand']
                        recommended_storage = product_info['storage']
                        
                        # Calculate expiry date using shelf life
                        expiry_date = manufacture_date + timedelta(days=shelf_life_days)
                        
                        # Show product information
                        st.success(f"✅ **{brand}** product found in database!")
                        st.info(f"**Product:** {item_name.title()}\n**Brand:** {brand}\n**Category:** {category}\n**Shelf Life:** {shelf_life_days} days\n**Recommended Storage:** {recommended_storage.title()}")
                        
                        # Warning if storage type doesn't match recommendation
                        if storage_type.lower() != recommended_storage:
                            st.warning(f"⚠️ **Storage Warning:** This product is typically stored in {recommended_storage.title()}, but you selected {storage_type.title()}. This may affect shelf life.")
                        
                    else:
                        # Product not in database - show error and suggestions
                        st.error("❌ **Product not found in database!**")
                        st.warning("This product is not available in our branded product database.")
                        
                        similar_products = branded_db.get_similar_products(item_name)
                        if similar_products:
                            st.info("**💡 Did you mean one of these products?**")
                            for product in similar_products[:5]:
                                st.text(f"• {product.title()}")
                        
                        st.error("Please enter a valid branded product name from our database.")
                        return  # Don't save the item
                    
                    # Save the item to database
                    conn = sqlite3.connect(db_path)
                    c = conn.cursor()
                    c.execute('''INSERT INTO items 
                                (name, category, manufacture_date, expiry_date, storage_type, alert_days_before) 
                                VALUES (?, ?, ?, ?, ?, ?)''',
                            (item_name, category, manufacture_date.strftime('%Y-%m-%d'), 
                             expiry_date.strftime('%Y-%m-%d'), storage_type.lower(), custom_alert_days))
                    conn.commit()
                    conn.close()
                    
                    st.success(f"✅ Item '{item_name}' saved successfully!")
                    st.info(f"**Predicted expiry date:** {expiry_date.strftime('%Y-%m-%d')}")
                else:
                    st.warning("⚠️ Please enter an item name.")
    
    elif menu_options[choice] == "upload_bill":
        st.markdown('<div class="sub-header">📷 Upload Grocery Bill</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image or PDF file", type=['jpg', 'jpeg', 'png', 'pdf'])
        
        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Bill", use_column_width=True)
                text = extract_text_from_image(image)
            else:
                text = extract_text_from_pdf(uploaded_file)
                st.success("PDF processed successfully")
            
            if text:
                with st.expander("📄 Extracted Text"):
                    st.text(text)
                
                manufacture_date, items = parse_receipt_text(text)
                
                st.subheader("📋 Extracted Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Manufacture Date:** {manufacture_date.strftime('%Y-%m-%d')}")
                with col2:
                    st.info(f"**Detected Items:** {len(items)}")
                
                if items:
                    st.subheader("💾 Save Detected Items")
                    storage_type = st.selectbox("Storage Type for all items", ["Pantry", "Refrigerator", "Freezer"])
                    bulk_alert_days = st.slider("Alert Days for all items", min_value=1, max_value=14, value=alert_days_before)
                    
                    if st.button("💾 Save All Items", use_container_width=True):
                        conn = sqlite3.connect(db_path)
                        c = conn.cursor()
                        
                        saved_count = 0
                        for item_info in items:
                            expiry_date = enhanced_predictor.predict_expiry(
                                item_info['name'], 
                                item_info['category'], 
                                manufacture_date, 
                                storage_type.lower()
                            )
                            
                            c.execute('''INSERT INTO items 
                                        (name, category, manufacture_date, expiry_date, storage_type, alert_days_before) 
                                        VALUES (?, ?, ?, ?, ?, ?)''',
                                    (item_info['name'], item_info['category'], 
                                     manufacture_date.strftime('%Y-%m-%d'), 
                                     expiry_date.strftime('%Y-%m-%d'), 
                                     storage_type.lower(), bulk_alert_days))
                            saved_count += 1
                        
                        conn.commit()
                        conn.close()
                        st.success(f"✅ Saved {saved_count} items successfully!")
    
    elif menu_options[choice] == "view_items":
        st.markdown('<div class="sub-header">👀 Your Grocery Items - Table View</div>', unsafe_allow_html=True)
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Get column names first to understand the structure
        c.execute("PRAGMA table_info(items)")
        columns_info = c.fetchall()
        
        # Clean up any corrupted data first
        c.execute("""
            UPDATE items 
            SET storage_type = CASE 
                WHEN storage_type LIKE '%-%' OR storage_type LIKE '%/%' THEN 'pantry'
                ELSE storage_type 
            END
            WHERE storage_type IS NOT NULL
        """)
        conn.commit()
        
        # Execute query with explicit column names to avoid confusion
        c.execute("""
            SELECT id, name, category, manufacture_date, expiry_date, storage_type, alert_days_before, notified 
            FROM items 
            ORDER BY expiry_date
        """)
        items = c.fetchall()
        
        if items:
            today = datetime.now().date()
            items_data = []
            
            for item in items:
                # Explicitly map each field to avoid confusion
                item_id = item[0]
                item_name = item[1]
                item_category = item[2]
                manufacture_date_str = item[3]
                expiry_date_str = item[4]
                storage_type = item[5]
                alert_days = item[6]
                notified = item[7]
                
                # Parse expiry date safely
                try:
                    expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
                    days_until_expiry = (expiry_date - today).days
                except (ValueError, TypeError):
                    expiry_date = today
                    days_until_expiry = 0
                
                # Determine status
                if days_until_expiry < 0:
                    status = "🔴 Expired"
                    days_text = f"Expired {abs(days_until_expiry)} days ago"
                elif days_until_expiry <= alert_days_before:
                    status = "🟡 Expiring Soon"
                    days_text = f"{days_until_expiry} days left"
                else:
                    status = "🟢 Safe"
                    days_text = f"{days_until_expiry} days left"
                
                # Create properly structured data
                items_data.append({
                    "ID": item_id,
                    "Name": item_name,
                    "Category": item_category,
                    "Manufacture Date": manufacture_date_str if manufacture_date_str else "Not set",
                    "Expiry Date": expiry_date_str if expiry_date_str else "Not set",
                    "Storage Type": storage_type.title() if storage_type else "Not set",
                    "Alert Days": alert_days,
                    "Days Until Expiry": days_text,
                    "Status": status,
                    "Notified": "✅ Yes" if notified else "❌ No"
                })
            
            df = pd.DataFrame(items_data)
            
            # Enhanced filtering and sorting options
            st.markdown("### 🔍 Filter & Sort Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_category = st.selectbox("📦 Filter by Category", ["All"] + sorted(list(df['Category'].unique())))
            with col2:
                filter_status = st.selectbox("📊 Filter by Status", ["All", "🟢 Safe", "🟡 Expiring Soon", "🔴 Expired"])
            with col3:
                filter_storage = st.selectbox("🏠 Filter by Storage", ["All"] + sorted(list(df['Storage Type'].unique())))
            with col4:
                sort_option = st.selectbox("📈 Sort by", ["Expiry Date (Ascending)", "Expiry Date (Descending)", "Name (A-Z)", "Name (Z-A)", "Category", "Storage Type"])
            
            # Apply filters
            filtered_df = df.copy()
            
            if filter_category != "All":
                filtered_df = filtered_df[filtered_df['Category'] == filter_category]
            
            if filter_status != "All":
                if filter_status == "🟢 Safe":
                    filtered_df = filtered_df[filtered_df['Status'] == "🟢 Safe"]
                elif filter_status == "🟡 Expiring Soon":
                    filtered_df = filtered_df[filtered_df['Status'] == "🟡 Expiring Soon"]
                elif filter_status == "🔴 Expired":
                    filtered_df = filtered_df[filtered_df['Status'] == "🔴 Expired"]
            
            if filter_storage != "All":
                filtered_df = filtered_df[filtered_df['Storage Type'] == filter_storage]
            
            # Apply sorting
            if sort_option == "Expiry Date (Ascending)":
                filtered_df = filtered_df.sort_values('Expiry Date')
            elif sort_option == "Expiry Date (Descending)":
                filtered_df = filtered_df.sort_values('Expiry Date', ascending=False)
            elif sort_option == "Name (A-Z)":
                filtered_df = filtered_df.sort_values('Name')
            elif sort_option == "Name (Z-A)":
                filtered_df = filtered_df.sort_values('Name', ascending=False)
            elif sort_option == "Category":
                filtered_df = filtered_df.sort_values('Category')
            elif sort_option == "Storage Type":
                filtered_df = filtered_df.sort_values('Storage Type')
            
            # Display results summary
            st.info(f"📊 **Showing {len(filtered_df)} of {len(df)} items**")
            
            # Display the filtered table
            if not filtered_df.empty:
                # Configure display columns in proper order
                display_columns = ['Name', 'Category', 'Manufacture Date', 'Expiry Date', 'Storage Type', 'Days Until Expiry', 'Status', 'Notified']
                
                # Create a clean display dataframe
                display_df = filtered_df[display_columns].copy()
                
                # Format the dates properly
                display_df['Manufacture Date'] = pd.to_datetime(display_df['Manufacture Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                display_df['Expiry Date'] = pd.to_datetime(display_df['Expiry Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                
                # Replace NaN values
                display_df = display_df.fillna('Not set')
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
                
                # Export options
                st.markdown("### 📤 Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV export
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv,
                        file_name=f"grocery_items_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Enhanced CSV export as primary option
                    # Create a formatted CSV with better structure
                    formatted_csv = display_df.to_csv(index=False, sep=',')
                    
                    st.download_button(
                        label="📊 Download as Enhanced CSV",
                        data=formatted_csv,
                        file_name=f"grocery_items_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        help="Download data as a formatted CSV file (opens in Excel)"
                    )
                    
                    # Try Excel export as secondary option
                    try:
                        import io
                        import openpyxl
                        
                        # Create Excel file
                        excel_df = display_df.copy()
                        output = io.BytesIO()
                        
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            excel_df.to_excel(writer, sheet_name='Grocery Items', index=False)
                            
                            # Basic formatting
                            workbook = writer.book
                            worksheet = writer.sheets['Grocery Items']
                            
                            # Auto-adjust column widths
                            for column in worksheet.columns:
                                max_length = 0
                                column_letter = column[0].column_letter
                                for cell in column:
                                    try:
                                        if len(str(cell.value)) > max_length:
                                            max_length = len(str(cell.value))
                                    except:
                                        pass
                                adjusted_width = min(max_length + 2, 50)
                                worksheet.column_dimensions[column_letter].width = adjusted_width
                        
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="📈 Download as Excel (Advanced)",
                            data=excel_data,
                            file_name=f"grocery_items_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download as Excel file with formatting"
                        )
                    except ImportError:
                        st.info("💡 **Excel Export:** Install `openpyxl` for Excel format")
                        st.code("pip install openpyxl")
                    except Exception as e:
                        st.info("💡 **Excel Export:** CSV format works great in Excel too!")
                
                # Management options
                st.markdown("### 🛠️ Item Management")
                
                if len(filtered_df) > 0:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Delete item
                        item_ids = filtered_df['ID'].tolist()
                        delete_id = st.selectbox(
                            "🗑️ Select item to delete",
                            options=item_ids,
                            format_func=lambda x: f"{filtered_df[filtered_df['ID'] == x]['Name'].iloc[0]} (ID: {x})"
                        )
                        
                        if st.button("🗑️ Delete Item", use_container_width=True):
                            c.execute("DELETE FROM items WHERE id = ?", (delete_id,))
                            conn.commit()
                            st.success("Item deleted successfully!")
                            st.rerun()
                    
                    with col2:
                        # Reset notifications
                        if st.button("🔄 Reset All Notifications", use_container_width=True):
                            c.execute("UPDATE items SET notified = 0")
                            conn.commit()
                            st.success("All notifications reset!")
                    
                    with col3:
                        # Bulk operations
                        st.markdown("**Bulk Operations:**")
                        if st.button("🗑️ Delete Expired Items", use_container_width=True):
                            expired_ids = filtered_df[filtered_df['Status'] == '🔴 Expired']['ID'].tolist()
                            if expired_ids:
                                c.execute(f"DELETE FROM items WHERE id IN ({','.join(map(str, expired_ids))})")
                                conn.commit()
                                st.success(f"Deleted {len(expired_ids)} expired items!")
                            else:
                                st.info("No expired items to delete.")
            else:
                st.info("No items match the selected filters. Try adjusting your filter criteria.")
                
                # Show suggestions
                if len(df) > 0:
                    st.markdown("**💡 Available categories:** " + ", ".join(df['Category'].unique()))
                    st.markdown("**💡 Available storage types:** " + ", ".join(df['Storage Type'].unique()))
        else:
            st.info("📝 No items saved yet. Add some items to get started!")
            st.markdown("""
            **Get started by:**
            1. 📦 Adding items manually using the "Manual Entry" option
            2. 📷 Uploading a grocery bill using the "Upload Bill" option
            """)
        
        conn.close()
    
    elif menu_options[choice] == "branded_products":
        st.markdown('<div class="sub-header">🏪 Available Branded Products Database</div>', unsafe_allow_html=True)
        
        st.info(f"💡 **This database contains {len(branded_db.products)} branded products with accurate shelf life information.**")
        
        # Enhanced search functionality with real-time suggestions
        search_term = st.text_input("🔍 Search for products", placeholder="e.g., Maggi, Cadbury, Amul...", key="branded_search")
        
        # Show search suggestions as user types
        if search_term and len(search_term) > 1:
            search_lower = search_term.lower()
            search_matches = []
            
            for product_key in branded_db.products.keys():
                if search_lower in product_key or search_lower in branded_db.products[product_key]['brand'].lower():
                    search_matches.append((product_key, branded_db.products[product_key]))
            
            if search_matches:
                st.info(f"💡 **Found {len(search_matches)} products matching '{search_term}'**")
                
                # Show quick suggestions without buttons (just display)
                quick_matches = search_matches[:5]
                for product_key, product_info in quick_matches:
                    st.text(f"• {product_key.title()} ({product_info['brand']}) - {product_info['category']} - {product_info['shelf_life_days']} days")
        
        # Filter by brand with search
        all_brands = sorted(list(set([product_info['brand'] for product_info in branded_db.products.values()])))
        selected_brand = st.selectbox("🏷️ Filter by Brand", ["All"] + all_brands)
        
        # Filter by category with search
        all_categories = sorted(list(set([product_info['category'] for product_info in branded_db.products.values()])))
        selected_category = st.selectbox("📦 Filter by Category", ["All"] + all_categories)
        
        # Quick category buttons
        st.markdown("**🎯 Quick Category Filters:**")
        quick_categories = ["Dairy", "Chocolates", "Biscuits", "Ready-to-Eat", "Beverages", "Snacks"]
        cols = st.columns(len(quick_categories))
        
        for i, category in enumerate(quick_categories):
            with cols[i]:
                if st.button(f"📦 {category}", key=f"branded_quick_cat_{category}"):
                    st.session_state.branded_search = ""
                    st.info(f"💡 Showing products from {category} category. Use the filters above to narrow down results.")
        
        # Filter products based on search and filters
        filtered_products = []
        for product_name, product_info in branded_db.products.items():
            # Search filter
            if search_term and search_term.lower() not in product_name.lower() and search_term.lower() not in product_info['brand'].lower():
                continue
            
            # Brand filter
            if selected_brand != "All" and product_info['brand'] != selected_brand:
                continue
            
            # Category filter
            if selected_category != "All" and product_info['category'] != selected_category:
                continue
            
            filtered_products.append((product_name, product_info))
        
        # Sort by brand, then by product name
        filtered_products.sort(key=lambda x: (x[1]['brand'], x[0]))
        
        st.info(f"📊 **Showing {len(filtered_products)} of {len(branded_db.products)} products**")
        
        if filtered_products:
            # Display products in a nice format
            for i, (product_name, product_info) in enumerate(filtered_products):
                with st.expander(f"🏷️ **{product_info['brand']}** - {product_name.title()}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**📦 Category:** {product_info['category']}")
                        st.write(f"**🏷️ Brand:** {product_info['brand']}")
                    
                    with col2:
                        st.write(f"**📅 Shelf Life:** {product_info['shelf_life_days']} days")
                        st.write(f"**🏠 Storage:** {product_info['storage'].title()}")
                    
                    with col3:
                        # Calculate expiry date example
                        example_manufacture_date = datetime.now()
                        example_expiry_date = example_manufacture_date + timedelta(days=product_info['shelf_life_days'])
                        st.write(f"**📆 Example:** If manufactured today, expires on {example_expiry_date.strftime('%Y-%m-%d')}")
                    
                    # Quick add button
                    if st.button(f"➕ Add {product_name.title()} to my items", key=f"add_{i}"):
                        st.success(f"💡 **Tip:** Use 'Manual Entry' to add '{product_name.title()}' with today's date as manufacture date!")
        else:
            st.info("No products match your search criteria. Try different search terms or filters.")
        
        # Statistics
        st.markdown("### 📊 Database Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand_counts = {}
            for product_info in branded_db.products.values():
                brand = product_info['brand']
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            
            st.write("**🏷️ Products by Brand:**")
            for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"• {brand}: {count} products")
        
        with col2:
            category_counts = {}
            for product_info in branded_db.products.values():
                category = product_info['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            st.write("**📦 Products by Category:**")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"• {category}: {count} products")
        
        with col3:
            storage_counts = {}
            for product_info in branded_db.products.values():
                storage = product_info['storage']
                storage_counts[storage] = storage_counts.get(storage, 0) + 1
            
            st.write("**🏠 Products by Storage:**")
            for storage, count in sorted(storage_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"• {storage.title()}: {count} products")
    
    elif menu_options[choice] == "dashboard":
        st.markdown('<div class="sub-header">📊 Expiry Analytics Dashboard</div>', unsafe_allow_html=True)
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Execute query with explicit column names to avoid confusion
        c.execute("""
            SELECT id, name, category, manufacture_date, expiry_date, storage_type, alert_days_before, notified 
            FROM items
        """)
        items = c.fetchall()
        conn.close()
        
        if items:
            items_data = []
            for item in items:
                # Explicitly map each field to avoid confusion
                item_id = item[0]
                item_name = item[1]
                item_category = item[2]
                manufacture_date_str = item[3]
                expiry_date_str = item[4]
                storage_type = item[5]
                alert_days = item[6]
                notified = item[7]
                
                items_data.append({
                    "ID": item_id,
                    "Name": item_name,
                    "Category": item_category,
                    "Manufacture Date": manufacture_date_str if manufacture_date_str else "Not set",
                    "Expiry Date": expiry_date_str if expiry_date_str else "Not set",
                    "Storage Type": storage_type.title() if storage_type else "Not set",
                    "Alert Days": alert_days,
                    "Notified": "Yes" if notified else "No"
                })
            
            df = pd.DataFrame(items_data)
            display_dashboard(df, alert_days_before)
        else:
            st.info("No data available for dashboard. Add some items first.")
    
    elif menu_options[choice] == "test_notification":
        st.markdown('<div class="sub-header">🔔 Test Notification System</div>', unsafe_allow_html=True)
        
        st.info("This will send a test email to verify your notification settings are working correctly.")
        
        if st.button("📧 Send Test Email", use_container_width=True):
            if user_email:
                test_body = f"""
                <html>
                <body>
                    <h2>Test Notification</h2>
                    <p>This is a test email from the Grocery Expiry Alert System.</p>
                    <p>If you received this, your email configuration is working correctly.</p>
                    <p>Your current alert preference is: {alert_days_before} day(s) before expiry.</p>
                    <br>
                    <p>Best regards,<br>Grocery Expiry Alert System Team</p>
                </body>
                </html>
                """
                
                if send_email(user_email, "Test Notification - Grocery Expiry Alert", test_body):
                    st.success("✅ Test email sent successfully!")
                else:
                    st.error("❌ Failed to send test email. Please check your email configuration.")
            else:
                st.error("No email address found for your account.")
    
    elif menu_options[choice] == "ml_training":
        st.markdown('<div class="sub-header">🤖 Machine Learning Model Training</div>', unsafe_allow_html=True)
        
        st.info("The ML model predicts expiry dates based on item category, storage conditions, and environmental factors.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            if enhanced_predictor.load_model():
                try:
                    with open('enhanced_expiry_model.pkl', 'rb') as f:
                        model_info = pickle.load(f)
                    st.success("✅ Model is loaded and ready!")
                    st.write(f"**R² Score:** {model_info.get('r2_score', 'N/A'):.3f}")
                    st.write(f"**MAE:** {model_info.get('mae', 'N/A'):.2f} days")
                    st.write(f"**Trained on:** {model_info.get('trained_at', 'N/A')}")
                except:
                    st.success("✅ Model is loaded!")
            else:
                st.warning("⚠️ No trained model found. Using enhanced fallback predictions.")
        
        with col2:
            st.subheader("Training Options")
            
            # Option to upload custom dataset
            st.write("Upload custom CSV data for training:")
            uploaded_data = st.file_uploader("Choose CSV file", type=['csv'], key="ml_upload")
            
            if st.button("🚀 Train with Enhanced Dataset", use_container_width=True):
                with st.spinner("Training model with realistic data..."):
                    success, message = enhanced_predictor.train_enhanced_model()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
            
            if uploaded_data is not None:
                if st.button("🚀 Train with Custom Data", use_container_width=True):
                    try:
                        df = pd.read_csv(uploaded_data)
                        st.write("Data preview:")
                        st.dataframe(df.head())
                        
                        with st.spinner("Training model with custom data..."):
                            success, message = enhanced_predictor.train_enhanced_model(df)
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
        
        # Feature importance visualization
        if enhanced_predictor.load_model() and hasattr(enhanced_predictor.model, 'feature_importances_'):
            st.subheader("📈 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': ['Category', 'Storage Type', 'Manufacture Month', 'Temperature', 'Humidity'],
                'Importance': enhanced_predictor.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance in Expiry Prediction",
                        color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    elif menu_options[choice] == "settings":
        st.markdown('<div class="sub-header">⚙️ System Settings</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Email Configuration")
            st.write("Configure your email settings in the .env file:")
            st.code("""
EMAIL_FROM=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
            """)
            
            # Test current configuration
            if st.button("Test Email Configuration", use_container_width=True):
                if os.getenv('EMAIL_FROM') and os.getenv('EMAIL_PASSWORD'):
                    st.success("✅ Email configuration found!")
                else:
                    st.error("❌ Email configuration missing!")
        
        with col2:
            st.subheader("System Status")
            status_items = [
                ("Database", os.path.exists(db_path), "Connected", "Not connected"),
                ("ML Model", enhanced_predictor.load_model(), "Loaded", "Using fallback"),
                ("Email Config", bool(os.getenv('EMAIL_FROM') and os.getenv('EMAIL_PASSWORD')), "Configured", "Not configured"),
                ("Scheduler", bool(os.getenv('EMAIL_FROM') and os.getenv('EMAIL_PASSWORD')), "Running", "Not running")
            ]
            
            for item_name, status, true_msg, false_msg in status_items:
                if status:
                    st.success(f"✅ **{item_name}:** {true_msg}")
                else:
                    st.warning(f"⚠️ **{item_name}:** {false_msg}")
    
    # Footer with information
    st.markdown("---")
    st.markdown("### 📋 How the System Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **1. Add Items**  
        ↳ Manual entry or bill upload
        """)
    
    with col2:
        st.markdown("""
        **2. ML Prediction**  
        ↳ AI predicts expiry using multiple factors
        """)
    
    with col3:
        st.markdown("""
        **3. Smart Monitoring**  
        ↳ System tracks expiry dates daily
        """)
    
    with col4:
        st.markdown("""
        **4. Timely Alerts**  
        ↳ Get notified before items expire
        """)

# Main function
# Main function
def main():
    # Initialize authentication
    if not init_auth():
        return
    
    # Initialize MongoDB
    db = init_mongodb()
    if db is None:
        st.error("Failed to connect to database. Please try again later.")
        return
    
    # Check if user is logged in
    user_id = cookies.get('user_id')
    user_email = cookies.get('user_email')
    user_name = cookies.get('user_name')
    
    # FIXED: Handle empty cookie value properly
    alert_days_cookie = cookies.get('alert_days_before')
    if alert_days_cookie and alert_days_cookie.strip():  # Check if not empty
        alert_days_before = int(alert_days_cookie)
    else:
        alert_days_before = 1  # Default value
    
    if user_id and user_email and user_name:
        main_app(db, user_id, user_email, user_name, alert_days_before)
    else:
        auth_page(db)

if __name__ == "__main__":
    main()