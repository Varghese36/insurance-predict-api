from flask import Flask, request, jsonify
import joblib

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
    app.run(debug=True)
