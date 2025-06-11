import numpy as np
import pickle
import os
from config import Config

# Load the trained Random Forest model
with open(Config.MODEL_PATH, "rb") as file:
    loaded_model = pickle.load(file)

# Load the saved LabelEncoder
with open(Config.ENCODER_PATH, "rb") as file:
    loaded_encoder = pickle.load(file)

# Get valid state labels
state_labels = loaded_encoder.classes_

def predict_profit(RnD_Spend, Administration, Marketing_Spend, State):
    # Encode 'State' column
    if State in state_labels:
        State_encoded = loaded_encoder.transform([State])[0]
    else:
        State_encoded = loaded_encoder.transform(["California"])[0]  # Default

    # Prepare input array
    input_features = np.array([[RnD_Spend, Administration, Marketing_Spend, State_encoded]])

    # Make Prediction
    predicted_profit = loaded_model.predict(input_features)[0]
    
    return predicted_profit
