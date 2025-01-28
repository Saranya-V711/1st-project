from flask import Flask, request,jsonify
import pandas
"""pip install flask"""
import pickle

with open("bank_marketing_model.pkl","rb") as f:
    model = pickle.load(f)
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data as JSON
    data = request.get_json()
    data = pandas.DataFrame(data)

# Make prediction
    predictions = model.predict(data)
    return jsonify({"bank_predictions": predictions.tolist()})

if __name__=="__main__":
    app.run(debug=True)
