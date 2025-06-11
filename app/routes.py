from flask import Blueprint, render_template, request
from app.predictor import predict_profit
import pickle
from config import Config

main = Blueprint("main", __name__)

# Load encoder for state options
with open(Config.ENCODER_PATH, "rb") as file:
    loaded_encoder = pickle.load(file)
state_labels = loaded_encoder.classes_

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            RnD_Spend = float(request.form["RnD_Spend"])
            Administration = float(request.form["Administration"])
            Marketing_Spend = float(request.form["Marketing_Spend"])
            State = request.form["State"]

            predicted_profit = predict_profit(RnD_Spend, Administration, Marketing_Spend, State)

            return render_template(
                "result.html",
                RnD_Spend=RnD_Spend,
                Administration=Administration,
                Marketing_Spend=Marketing_Spend,
                State=State,
                predicted_profit=predicted_profit,
            )
        except Exception as e:
            return f"Error: {e}"

    return render_template("predict.html", states=state_labels)

