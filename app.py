from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load Random Forest model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature names for reference
FEATURE_NAMES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Check if features are provided
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        # Convert input to numpy array
        input_features = np.array(data["features"]).reshape(1, -1)
        
        # Validate feature count
        if input_features.shape[1] != 9:
            return jsonify({
                "error": f"Expected 9 features, got {input_features.shape[1]}",
                "required_features": FEATURE_NAMES
            }), 400
        
        # Predict using the model
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)
        
        return jsonify({
            "prediction": int(prediction[0]),
            "potability": "Safe to drink" if prediction[0] == 1 else "Not safe to drink",
            "confidence": {
                "not_potable": float(probability[0][0]),
                "potable": float(probability[0][1])
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=True)