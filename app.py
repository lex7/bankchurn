from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import joblib

app = Flask(__name__)

# загрузка предобученной модели CatBoost
catboost_model = CatBoostClassifier()
catboost_model.load_model("cb_model.json")

# загрузка обученного Min-Max Scaler и One-Hot Encoder
scaler = joblib.load('min_max_scaler.save')
encoder = joblib.load('one_hot_encoder.save')

# функция предобработки
def preprocess_data(data):

    # Перевод строк в целые числа
    numeric_fields = ['Age', 'NumOfProducts', 'CreditScore', 'EstimatedSalary', 'Balance', 'Tenure']
    for field in numeric_fields:
        data[field] = data[field].astype(int)

    # Генерация признака
    data['TargetAge'] = (56 - np.abs(data['Age'] - 56)) / 56

    # One-Hot Encoding
    columns_to_encode = ['Geography', 'Gender', 'NumOfProducts']
    encoded_data = encoder.transform(data[columns_to_encode])

    encoded_columns = [
        'Geography_France','Geography_Germany','Geography_Spain', 
        'Gender_Male',
        'NumOfProducts_1','NumOfProducts_2','NumOfProducts_3','NumOfProducts_4'
    ]
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)
    data = pd.concat([data, encoded_df], axis=1)
    data.drop(columns=columns_to_encode, inplace=True)
    data.drop(columns='Geography_Spain', inplace=True)

    # Упорядочивание колонок
    ordered_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'IsActiveMember',
        'EstimatedSalary', 'TargetAge', 'Geography_France', 'Geography_Germany',
        'Gender_Male', 'NumOfProducts_1', 'NumOfProducts_2', 'NumOfProducts_3',
        'NumOfProducts_4'
    ]
    data = data[ordered_columns]

    # Min-Max Scaling
    features = data.columns
    print(features)
    data[features] = scaler.transform(data[features])
    
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print(data)
    preprocessed_data = preprocess_data(pd.DataFrame([data]))
    
    prediction_proba = catboost_model.predict_proba(preprocessed_data)
    prediction = catboost_model.predict(preprocessed_data)

    # Перевод предсказаний
    prediction_label = 'Останется' if prediction[0] == 0 else 'Покинет'

    # Уверенность в предсказании
    certainty = max(prediction_proba[0])
    
    return jsonify({'prediction': prediction_label, 'certainty': certainty})
    
if __name__ == '__main__':
    app.run(debug=True)
