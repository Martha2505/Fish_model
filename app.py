from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load saved model, scaler, and encoder
model = pickle.load(open("fish_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = [float(x) for x in request.form.values()]
        transformed_data = scaler.transform([data])
        
        # Predict
        prediction = model.predict(transformed_data)
        species = encoder.inverse_transform(prediction)[0]
        
        return jsonify({"Prediction": species})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
 
