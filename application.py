from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load files
model = pickle.load(open("models/ridge.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
cols = pickle.load(open("models/columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("idex.html")


@app.route("/predict", methods=["POST"])
def predict():
    airline = request.form["airline"]
    source = request.form["source"]
    dest = request.form["destination"]
    stops = float(request.form["total_stops"])
    duration = float(request.form["duration_hours"])

    # Blank row
    input_dict = dict.fromkeys(cols, 0)

    # Set numeric
    input_dict["total_Stops"] = stops
    input_dict["duration_hours"] = duration

    # Set one-hot
    input_dict[f"Airline_{airline}"] = 1
    input_dict[f"Source_{source}"] = 1
    input_dict[f"Destination_{dest}"] = 1

    df = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]

    return render_template("idex.html", prediction=round(pred,2))


if __name__ == "__main__":
    app.run(debug=True)
