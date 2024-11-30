from tensorflow.keras.models import load_model

# Charger le modèle Keras au format .h5
crop_recommendation_model_path = './models/DeepNN.h5'
crop_recommendation_model = load_model(crop_recommendation_model_path)

def topCrops(crops, data):
    # Utilisation de la méthode predict pour obtenir les probabilités des cultures
    probs = crop_recommendation_model.predict(data)
    
    # Dictionnaire pour stocker les probabilités des cultures
    crop_probs = {}
    for i in range(len(crops)):
        crop_probs[crops[i]] = probs[i][0]  # Ici, on suppose que probs[i] est un vecteur avec les scores pour chaque culture
    
    # Trier les cultures par probabilité décroissante et garder les 5 meilleures
    top_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    top_crops = dict(top_crops)  # Transformer en dictionnaire

    return top_crops
