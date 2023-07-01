from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("heart.csv")

# Preprocess the data
def preprocess_data():
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    # Perform label encoding for categorical variables
    categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'MaxHR', 'ST_Slope', 'Cholesterol', 'ExerciseAngina']
    for feature in categorical_features:
        label_encoder = LabelEncoder()
        X[feature] = label_encoder.fit_transform(X[feature])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Train the model
def train_model(X, y):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X, y)
    return rf_classifier

# Initialize data preprocessing and model training
X_scaled, y, scaler = preprocess_data()
rf_classifier = train_model(X_scaled, y)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)

# Load the model
with open('model.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form inputs
        Age = float(request.form['Age'])
        Sex = int(request.form['Sex'])
        ChestPainType = int(request.form['ChestPainType'])
        RestingBP = float(request.form['RestingBP'])
        Cholesterol = float(request.form['Cholesterol'])
        FastingBS = int(request.form['FastingBS'])
        RestingECG = int(request.form['RestingECG'])
        MaxHR = float(request.form['MaxHR'])
        ExerciseAngina = int(request.form['ExerciseAngina'])
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = int(request.form['ST_Slope'])
   
        # Create a feature vector
        feature_vector = np.array([Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope])

        # Scale the feature vector
        scaled_feature_vector = scaler.transform([feature_vector])

        # Make the prediction
        prediction = rf_classifier.predict(scaled_feature_vector)

        # Render the result on a new page
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        logging.error(str(e))
        return render_template('error.html')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
