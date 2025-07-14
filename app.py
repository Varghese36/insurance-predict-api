from flask import Flask, request, jsonify
import joblib
import os


app = Flask(__name__)

# Load your saved model
model = joblib.load("insurance_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['age'],
        data['sex'],
        data['bmi'],
        data['children'],
        data['smoker'],
        data['region']
    ]
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
