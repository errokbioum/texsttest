
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from weather import weather_fetch
from rain import rain_info
from plot import topCrops
import os
from weather import weather_fetch
from rain import rain_info
from info import info_range
import json
import pandas as pd
from flask_cors import CORS
rainfall_data = rain_info()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
app = Flask(__name__)
CORS(app) 
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



crop_recommendation_model_path = './models/DeepNN.h5'
crop_recommendation_model = load_model(crop_recommendation_model_path)

# Chargement des cultures possibles
crops = np.load('crops.npy', allow_pickle=True)  # Assurez-vous que le chemin du fichier crops.npy est correct

# Initialisation de l'application Flask
app = Flask(__name__)

# Load rainfall data and the DNN model
rainfall_data = rain_info()
crop_yield_model_path = "./models/Yield_DNN.h5"  # Path to the new DNN model
crop_yield_model = load_model(crop_yield_model_path)

# Load the columns used in the original training
with open('columns.json', 'r') as f:
    data = json.load(f)


def crop_yield(formdata):
    crop = formdata["crop"]
    area = int(formdata["area"])
    season = formdata["season"]
    city = formdata["city"]

    # Fetch weather information
    temperature, humidity = weather_fetch(city)

    # Get rainfall data for the city and season
    rainfall = rainfall_data[rainfall_data["DIST"] == city][season].values[0]

    # Prepare the input DataFrame for the model
    columns = [index for index in data]  # Ensure the columns are in the correct order
    df = pd.DataFrame(columns=columns)

    df.loc[0] = 0  # Initialize all values to zero

    # Set input feature values
    df["Year"] = 2016  # Ensure this column matches the trained model
    if city in df.columns:
        df[city] = 1
    if season in df.columns:
        df[season] = 1
    if crop in df.columns:
        df[crop] = 1
    df["Area"] = area
    df["Temperature"] = temperature
    df["Rainfall"] = rainfall

    # Debugging step: Print columns and data types to ensure everything is correct
    print(f"Colonnes du DataFrame : {df.columns}")
    print(f"Types des colonnes : {df.dtypes}")
    
    # Align columns with those expected by the model (make sure the correct number of columns)
    df = df[columns]  # Reorder and ensure no extra columns

    # Check the shape of the DataFrame before predicting
    print(f"Shape du DataFrame : {df.shape}")

    input_features = df.values.astype(np.float32)  # Convert to numpy array

    # Debugging step: Print the shape of the input features
    print(f"Input shape for the model: {input_features.shape}")

    # Make prediction using the DNN model
    my_prediction = crop_yield_model.predict(input_features)
    
    # Convert the prediction to float before returning
    prediction = float(my_prediction[0][0])  # Ensure it's a float

    return prediction, temperature, humidity, rainfall

# Route pour la prédiction du rendement des cultures
@app.route('/crop-yield-predict', methods=['POST'])
def crop_yield_prediction():
    data = request.json
    formdata = data['formdata']
    
    # Obtenir la prédiction
    prediction, temperature, humidity, rainfall = crop_yield(formdata)
    
    # Traitement des données de l'année et de la saison
    rainfall = round(rainfall, 2)
    year_yield, season_yield, temp_yield, rain_yield, humid_yield = info_range(formdata, temperature, humidity, rainfall)
    year_yield[2022] = round(prediction / int(formdata['area']), 2)
    
    # Préparer la réponse
    pred = {
        "prediction": prediction,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "year_yield": year_yield,
        "season_yield": season_yield,
        "temp_yield": temp_yield,
        "rain_yield": rain_yield,
        "humid_yield": humid_yield,
    }
    
    # Si aucune culture ne peut être cultivée
    if pred == '':
        response = {
            "status": "error",
            "result": pred,
            "message": "No crop can be grown in this region"
        }
    else:
        response = {
            "status": "success",
            "result": pred,
            "message": "Crop Yield fetched successfully"
        }
    
    return jsonify(response)


# Fonction de prédiction de culture
def crop_recommendation(formdata):
    rainfall_data = rain_info()
    N = formdata['nitrogen']
    P = formdata['phosphorous']
    K = formdata['pottasium']
    ph = formdata['ph']
    season = formdata['season']
    city = formdata['city']
    temperature, humidity = weather_fetch(city)
    rainfall = rainfall_data[rainfall_data["DIST"] == city][season].values[0]
    data = [[N, P, K, temperature, humidity, ph, rainfall]]
    my_prediction = crop_recommendation_model.predict(data)
    prediction = []
    for i in range(0, len(my_prediction[0])):
        if my_prediction[0][i] == 1:
            prediction.append(crops[i])
    if len(prediction) == 0:
        prediction = ['No crop']
    chart_data = topCrops(crops, data)
    return prediction, temperature, humidity, rainfall, chart_data
    
# Route pour la prédiction des cultures
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    data = request.get_json()  # Récupère les données JSON envoyées par le client
    formdata = data.get('formdata')  # Récupère les données du formulaire
    if not formdata:
        return jsonify({"status": "error", "message": "Missing form data"}), 400

    # Appel de la fonction de recommandation de culture
    prediction = crop_recommendation(formdata)

    # Préparation de la réponse
    if prediction == ['No crop']:
        response = {
            "status": "error",
            "result": prediction,
            "message": "No crop can be grown in this region"
        }
    else:
        response = {
            "status": "success",
            "result": prediction,
            "message": "Crop recommendation fetched successfully"
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

