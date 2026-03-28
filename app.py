from flask import Flask, request, render_template
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load trained model (Pipeline)
model = joblib.load("model.pkl")

# Load feature info (not used for encoding, just reference)
with open("feature_order.json", "r") as f:
    feature_info = json.load(f)

categorical_cols = feature_info["categorical"]
numeric_cols = feature_info["numeric"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Read input from HTML form
    company = request.form.get("company")
    wb = float(request.form.get("work_balance_score"))
    cv = float(request.form.get("culture_values_score"))
    cb = float(request.form.get("compensation_benefit_score"))
    co = float(request.form.get("career_opportunities_score"))
    sm = float(request.form.get("senior_management_score"))

    # Structure data exactly how the model expects it
    input_data = np.array([[company, wb, cv, cb, co, sm]])

    # Predict using the pipeline
    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Employee Rating: {prediction:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
