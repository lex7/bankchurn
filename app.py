from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load the pre-trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")

# Load the pre-fitted MinMaxScaler
scaler = joblib.load('min_max_scaler.save') 
encoder = joblib.load('one_hot_encoder.save')

# Define preprocessing function
def preprocess_data(data):
    # Convert 'Age' field to integer
    data['Age'] = data['Age'].astype(int)

    # Preprocess the data similar to how you did before training
    data['TargetAge'] = (56 - np.abs(data['Age'] - 56)) / 56 
    
    # One-Hot encoding
    encoded_data = encoder.transform(data[['Geography', 'Gender', 'NumOfProducts']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Geography', 'Gender', 'NumOfProducts']), index=data.index)
    data = pd.concat([data, encoded_df], axis=1)
    data.drop(columns=[
        'Geography',
        'Gender',
        'NumOfProducts',
        
        ], inplace=True)
    
    # Transform data using the pre-fitted scaler
    features = data.columns
    data[features] = scaler.transform(data[features])
    
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.form.to_dict()

    print(data)
    
    # Preprocess the input data
    preprocessed_data = preprocess_data(pd.DataFrame([data]))
    
    # Make prediction
    prediction = xgb_model.predict(xgb.DMatrix(preprocessed_data))
    
    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)