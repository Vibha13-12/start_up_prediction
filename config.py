import os

class Config:
    SECRET_KEY = "your_secret_key"
    MODEL_PATH = os.path.join("app", "models", "random_forest_model.sav")
    ENCODER_PATH = os.path.join("app", "models", "label_encoder.pkl")
