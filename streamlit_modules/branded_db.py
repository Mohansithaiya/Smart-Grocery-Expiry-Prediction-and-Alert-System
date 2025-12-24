# Branded product shelf-life database

class BrandedProductDatabase:
    def __init__(self):
        self.products = {
            'maggi noodles': {'category': 'Ready-to-Eat', 'shelf_life_days': 80, 'brand': 'Maggi', 'storage': 'pantry'},
            'cadbury dairy milk': {'category': 'Chocolates', 'shelf_life_days': 180, 'brand': 'Cadbury', 'storage': 'pantry'},
            'amul butter': {'category': 'Dairy', 'shelf_life_days': 90, 'brand': 'Amul', 'storage': 'refrigerator'},
            'parle-g': {'category': 'Biscuits', 'shelf_life_days': 180, 'brand': 'Parle', 'storage': 'pantry'},
            'lays chips': {'category': 'Snacks', 'shelf_life_days': 90, 'brand': 'Lays', 'storage': 'pantry'},
            'coca cola': {'category': 'Beverages', 'shelf_life_days': 180, 'brand': 'Coca Cola', 'storage': 'pantry'},
            'milk': {'category': 'Dairy', 'shelf_life_days': 3, 'brand': 'Generic', 'storage': 'refrigerator'}
        }

    def get_product_info(self, product_name):
        product_name = product_name.lower().strip()
        for key, value in self.products.items():
            if key in product_name or product_name in key:
                return value
        return None

    def get_similar_products(self, product_name):
        product_name = product_name.lower()
        return [k for k in self.products if product_name in k]


branded_db = BrandedProductDatabase()


def get_autocomplete_suggestions(query, limit=5):
    if not query:
        return []
    query = query.lower()
    results = []
    for product, info in branded_db.products.items():
        if query in product:
            results.append({
                'product': product,
                'brand': info['brand'],
                'category': info['category'],
                'shelf_life': info['shelf_life_days']
            })
    return results[:limit]
