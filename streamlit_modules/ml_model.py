# Machine learning expiry prediction logic

import numpy as np
import pickle
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler


class EnhancedExpiryPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fallback_prediction(self, manufacture_date, days=7):
        return manufacture_date + timedelta(days=days)

    def load_model(self):
        try:
            with open('enhanced_expiry_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.label_encoders = data['encoders']
            return True
        except:
            return False

    def predict_expiry(self, manufacture_date):
        return self.fallback_prediction(manufacture_date)


enhanced_predictor = EnhancedExpiryPredictor()
