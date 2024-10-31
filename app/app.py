from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load all the necessary files
def load_files():
    model_dir = Path("model")
    
    try:
        # Load model and components using joblib
        model = joblib.load(model_dir / "best_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        label_encoders = joblib.load(model_dir / "label_encoders.pkl")
        categorical_columns = joblib.load(model_dir / "categorical_columns.pkl")
        columns_order = joblib.load(model_dir / "column_order.pkl")

        # Load model info
        with open(model_dir / "model_info.txt", "r") as f:
            model_info = f.read().strip()
            
        return model, scaler, label_encoders, categorical_columns, model_info,columns_order
    except Exception as e:
        print(f"Error loading model files: {str(e)}")
        raise

# Load all components
model, scaler, label_encoders, categorical_columns, model_info ,columns_order= load_files()

# Define the feature groups
numeric_columns = ['HeightInMeters', 'WeightInKilograms', 'BMI']
binary_fields = [
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
    'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
    'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'HighRiskLastYear', 'CovidPos'
]

# Define form options (your existing form_options dictionary here)
form_options = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'California', 'Arkansas', 'Connecticut',
              'Colorado', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Indiana',
              'District of Columbia', 'Kansas', 'Iowa', 'Maryland', 'Minnesota', 'Montana',
              'New Jersey', 'Nebraska', 'New York', 'Ohio', 'Oklahoma', 'Texas', 'Vermont',
              'Washington', 'Utah', 'Illinois', 'West Virginia', 'Virginia', 'Massachusetts',
              'Kentucky', 'Louisiana', 'Maine', 'Wisconsin', 'Michigan', 'Mississippi',
              'Missouri', 'Nevada', 'New Hampshire', 'New Mexico', 'South Carolina',
              'North Carolina', 'North Dakota', 'Oregon', 'Pennsylvania', 'Rhode Island',
              'South Dakota', 'Tennessee', 'Wyoming', 'Guam', 'Puerto Rico', 'Virgin Islands'],
    'Sex': ['Female', 'Male'],
    'GeneralHealth': ['Fair', 'Very good', 'Excellent', 'Good', 'Poor'],
    'AgeCategory': ['Age 75 to 79', 'Age 65 to 69', 'Age 60 to 64', 'Age 70 to 74',
                    'Age 50 to 54', 'Age 80 or older', 'Age 55 to 59', 'Age 25 to 29',
                    'Age 40 to 44', 'Age 30 to 34', 'Age 35 to 39', 'Age 18 to 24',
                    'Age 45 to 49'],
    'HadDiabetes': ['Yes', 'No', 'No, pre-diabetes or borderline diabetes',
                    'Yes, but only during pregnancy (female)'],
    'SmokerStatus': ['Former smoker', 'Never smoked', 'Current smoker - now smokes every day',
                     'Current smoker - now smokes some days'],
    'ECigaretteUsage': ['Never used e-cigarettes in my entire life', 'Not at all (right now)',
                        'Use them some days', 'Use them every day'],
    'RaceEthnicityCategory': ['White only, Non-Hispanic', 'Black only, Non-Hispanic',
                             'Other race only, Non-Hispanic', 'Multiracial, Non-Hispanic', 'Hispanic'],
    'TetanusLast10Tdap': ['No, did not receive any tetanus shot in the past 10 years',
                          'Yes, received Tdap', 'Yes, received tetanus shot but not sure what type',
                          'Yes, received tetanus shot, but not Tdap']
}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', 
                         form_options=form_options,
                         binary_fields=binary_fields,
                         model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create DataFrame with a single row
        df = pd.DataFrame([data])
        
        # Convert numeric fields
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert binary fields to numeric (0/1)
        for col in binary_fields:
            df[col] = (df[col] == 'Yes').astype(int)
        
        # Apply label encoding to categorical columns
        for col in categorical_columns:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])
        
        # Create feature list in the correct order
        all_features = columns_order
        
        # Debug prints
        print("Expected features:", all_features)
        print("DataFrame columns:", df.columns.tolist())


        print("Prediction features:", df.columns.tolist())
        
        # Reorder columns to match training data
        df = df[all_features]
        
        # Apply scaling
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0][1]),
            'model_type': model_info
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Log the full error
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)