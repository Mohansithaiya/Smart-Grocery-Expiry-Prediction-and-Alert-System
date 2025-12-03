import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create comprehensive dataset for training
def create_training_dataset():
    """Create a comprehensive dataset for ML model training"""
    
    # Branded products data
    branded_products = {
        # Chocolates
        'Cadbury Dairy Milk Chocolate': {'category': 'Chocolates', 'shelf_life_days': 270, 'brand': 'Cadbury'},
        'Nestle KitKat': {'category': 'Chocolates', 'shelf_life_days': 365, 'brand': 'Nestle'},
        'Amul Dark Chocolate': {'category': 'Chocolates', 'shelf_life_days': 180, 'brand': 'Amul'},
        'Cadbury 5 Star': {'category': 'Chocolates', 'shelf_life_days': 300, 'brand': 'Cadbury'},
        'Toblerone': {'category': 'Chocolates', 'shelf_life_days': 365, 'brand': 'Mondelez'},
        
        # Cooking Oils
        'Fortune Sunflower Oil': {'category': 'Cooking Oil', 'shelf_life_days': 540, 'brand': 'Fortune'},
        'Saffola Gold Oil': {'category': 'Cooking Oil', 'shelf_life_days': 365, 'brand': 'Saffola'},
        'Patanjali Mustard Oil': {'category': 'Cooking Oil', 'shelf_life_days': 720, 'brand': 'Patanjali'},
        
        # Ready-to-Eat
        'MTR Ready To Eat Pongal': {'category': 'Ready-to-Eat', 'shelf_life_days': 365, 'brand': 'MTR'},
        'Haldiram Bhujia': {'category': 'Ready-to-Eat', 'shelf_life_days': 180, 'brand': 'Haldiram'},
        'Britannia Cake': {'category': 'Ready-to-Eat', 'shelf_life_days': 90, 'brand': 'Britannia'},
        
        # Noodles
        'Maggi Noodles': {'category': 'Noodles', 'shelf_life_days': 270, 'brand': 'Nestle'},
        'Top Ramen Noodles': {'category': 'Noodles', 'shelf_life_days': 240, 'brand': 'Nissin'},
        'Sunfeast Yippee Noodles': {'category': 'Noodles', 'shelf_life_days': 300, 'brand': 'Sunfeast'},
        
        # Beverages
        'Bru Instant Coffee': {'category': 'Beverages', 'shelf_life_days': 365, 'brand': 'HUL'},
        'Tata Tea Gold': {'category': 'Beverages', 'shelf_life_days': 540, 'brand': 'Tata'},
        'Red Label Tea': {'category': 'Beverages', 'shelf_life_days': 365, 'brand': 'Brooke Bond'},
        
        # Dairy
        'Amul Butter': {'category': 'Dairy', 'shelf_life_days': 180, 'brand': 'Amul'},
        'Nestle Milkmaid': {'category': 'Dairy', 'shelf_life_days': 365, 'brand': 'Nestle'},
        'Mother Dairy Curd': {'category': 'Dairy', 'shelf_life_days': 7, 'brand': 'Mother Dairy'},
        
        # Biscuits
        'Parle-G Biscuits': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Parle'},
        'Britannia Good Day': {'category': 'Biscuits', 'shelf_life_days': 150, 'brand': 'Britannia'},
        'Oreo Cookies': {'category': 'Biscuits', 'shelf_life_days': 365, 'brand': 'Mondelez'},
        
        # Masala
        'Everest Garam Masala': {'category': 'Masala', 'shelf_life_days': 365, 'brand': 'Everest'},
        'MDH Chana Masala': {'category': 'Masala', 'shelf_life_days': 360, 'brand': 'MDH'},
        'Catch Turmeric Powder': {'category': 'Masala', 'shelf_life_days': 365, 'brand': 'Catch'},
        
        # Instant Mix
        'MTR Rava Idli Mix': {'category': 'Instant Mix', 'shelf_life_days': 180, 'brand': 'MTR'},
        'Gits Dhokla Mix': {'category': 'Instant Mix', 'shelf_life_days': 270, 'brand': 'Gits'},
        'MTR Masala Oats': {'category': 'Instant Mix', 'shelf_life_days': 365, 'brand': 'MTR'},
    }
    
    data = []
    np.random.seed(42)
    
    # Add branded products data
    for product_name, details in branded_products.items():
        for month in range(1, 13):
            # Seasonal effects
            seasonal_factor = 1.0
            if month in [6, 7, 8]:  # Summer
                seasonal_factor = 0.9
            elif month in [12, 1, 2]:  # Winter
                seasonal_factor = 1.1
            
            # Storage types
            for storage in ['pantry', 'refrigerator', 'freezer']:
                # Environmental factors based on storage
                if storage == 'pantry':
                    temp = np.random.normal(25, 3)
                    humidity = np.random.normal(60, 8)
                elif storage == 'refrigerator':
                    temp = np.random.normal(4, 1)
                    humidity = np.random.normal(80, 5)
                else:  # freezer
                    temp = np.random.normal(-18, 2)
                    humidity = np.random.normal(20, 5)
                
                # Calculate shelf life
                base_shelf_life = details['shelf_life_days']
                shelf_life = base_shelf_life * seasonal_factor
                shelf_life += np.random.normal(0, shelf_life * 0.05)
                shelf_life = max(1, int(shelf_life))
                
                data.append({
                    'item_name': product_name,
                    'category': details['category'],
                    'brand': details['brand'],
                    'manufacture_month': month,
                    'storage_type': storage,
                    'temperature': temp,
                    'humidity': humidity,
                    'season': 'summer' if month in [6, 7, 8] else 'winter' if month in [12, 1, 2] else 'other',
                    'shelf_life_days': shelf_life,
                    'is_branded': 1
                })
    
    # Add generic food categories
    generic_categories = {
        'Dairy': {'pantry': 2, 'refrigerator': 14, 'freezer': 90},
        'Meat': {'pantry': 1, 'refrigerator': 5, 'freezer': 180},
        'Fruits': {'pantry': 7, 'refrigerator': 14, 'freezer': 60},
        'Vegetables': {'pantry': 5, 'refrigerator': 10, 'freezer': 90},
        'Bakery': {'pantry': 5, 'refrigerator': 7, 'freezer': 30},
        'Grains': {'pantry': 365, 'refrigerator': 365, 'freezer': 365},
        'Beverages': {'pantry': 180, 'refrigerator': 180, 'freezer': 180}
    }
    
    generic_items = {
        'Dairy': ['milk', 'yogurt', 'cheese', 'butter', 'cream'],
        'Meat': ['chicken', 'beef', 'pork', 'fish', 'lamb'],
        'Fruits': ['apples', 'bananas', 'oranges', 'berries', 'grapes'],
        'Vegetables': ['lettuce', 'tomatoes', 'carrots', 'broccoli', 'spinach'],
        'Bakery': ['bread', 'cakes', 'pastries', 'buns', 'cookies'],
        'Grains': ['rice', 'pasta', 'cereal', 'flour', 'oats'],
        'Beverages': ['juice', 'soda', 'water', 'tea', 'coffee']
    }
    
    for category, storage_life in generic_categories.items():
        for item in generic_items[category]:
            for month in range(1, 13):
                for storage, base_life in storage_life.items():
                    # Seasonal effects
                    seasonal_factor = 1.0
                    if month in [6, 7, 8]:  # Summer
                        if storage == 'pantry':
                            seasonal_factor = 0.7
                    elif month in [12, 1, 2]:  # Winter
                        seasonal_factor = 1.1
                    
                    # Environmental factors
                    if storage == 'pantry':
                        temp = np.random.normal(25, 5)
                        humidity = np.random.normal(60, 10)
                    elif storage == 'refrigerator':
                        temp = np.random.normal(4, 1)
                        humidity = np.random.normal(80, 5)
                    else:  # freezer
                        temp = np.random.normal(-18, 2)
                        humidity = np.random.normal(20, 5)
                    
                    # Calculate shelf life
                    shelf_life = base_life * seasonal_factor
                    shelf_life += np.random.normal(0, shelf_life * 0.1)
                    shelf_life = max(1, int(shelf_life))
                    
                    data.append({
                        'item_name': item,
                        'category': category,
                        'brand': 'generic',
                        'manufacture_month': month,
                        'storage_type': storage,
                        'temperature': temp,
                        'humidity': humidity,
                        'season': 'summer' if month in [6, 7, 8] else 'winter' if month in [12, 1, 2] else 'other',
                        'shelf_life_days': shelf_life,
                        'is_branded': 0
                    })
    
    df = pd.DataFrame(data)
    df.to_csv('grocery_expiry_dataset.csv', index=False)
    print(f"Dataset created with {len(df)} samples")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Brands: {df['brand'].nunique()}")
    print(f"Sample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    create_training_dataset()
