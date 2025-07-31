from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model
with open('rf_classifier_selected.pkl', 'rb') as f:
    rf_classifier_selected = pickle.load(f)

# Define selected features
selected_features = ['hemo', 'pcv', 'sg', 'sc', 'al', 'rbc', 'rc', 'dm', 'bgr', 'sod', 'htn', 'bu', 'pc']

# Define prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get attribute values from the form
        attributes = []
        for feature in selected_features:
            value = float(request.form[feature])
            attributes.append(value)
        
        # Make prediction using the trained model
        strength, disease_stage = predict_disease_stage(attributes, rf_classifier_selected)
        
        # Return prediction result
        return render_template('result.html', selected_features=selected_features, strength=strength, disease_stage=disease_stage)
    
    # Render the form template for GET requests
    return render_template('index.html', selected_features=selected_features)

def predict_disease_stage(attributes, rf_classifier):
    # Create a feature array
    features = np.array([attributes])
    
    # Make prediction using the trained model
    prediction_prob = rf_classifier.predict_proba(features)
    
    # Get the probability of having the disease (class 1)
    disease_probability = prediction_prob[0][1]
    
    # Classify into low-level and high-level based on the threshold
    """"if disease_probability <= 0.5:
        return 0, "No Disease"  # Strength 0 for no disease
    elif disease_probability >= 0.7:
        return 1, "Low-level Disease"  # Strength 1 for disease, with indication of level
    else:
        return 1, "High-level Disease"  # Strength 1 for disease, with indication of level"""
    if disease_probability <= 0.5:
        return 0, "No Disease"  # Strength 0 for no disease
    else:
        return 1, "High-level Disease" if disease_probability >= 0.7 else "Low-level Disease"

if __name__ == '__main__':
    app.run(debug=True)

