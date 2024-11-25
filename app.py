import json
import base64
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os

# Cargar el archivo JSON
def load_data(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"El archivo {json_file} no existe.")
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Decodificar una imagen base64
def decode_image(base64_str):
    try:
        img_data = base64.b64decode(base64_str.split(',')[1])
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        print(f"Error al decodificar la imagen: {e}")
        return None

# Convertir una imagen en una representación de características (e.g., tamaño)
def extract_features(image):
    return [image.size[0], image.size[1]]  # Ancho y alto como características

# Preparar los datos para el modelo
def prepare_data(data):
    features = []
    labels = []
    for label, images in data['data'].items():
        for img_base64 in images:
            img = decode_image(img_base64)
            if img:
                features.append(extract_features(img))
                labels.append(label)
    return np.array(features), np.array(labels)

# Entrenar el modelo
def train_model(features, labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Modelo entrenado con una precisión del {accuracy * 100:.2f}%")
    return model, le

# Predecir el tamaño de un nuevo objeto
def predict(model, label_encoder, img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"La imagen {img_path} no existe.")
    img = Image.open(img_path)
    features = extract_features(img)
    prediction = model.predict([features])
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

if __name__ == "__main__":
    json_file = "data.json"
    model_file = "model.pkl"

    if not os.path.exists(model_file):
        # Cargar y preparar los datos
        data = load_data(json_file)
        features, labels = prepare_data(data)

        # Entrenar el modelo
        model, label_encoder = train_model(features, labels)

        # Guardar el modelo entrenado
        joblib.dump((model, label_encoder), model_file)
    else:
        # Cargar el modelo guardado
        model, label_encoder = joblib.load(model_file)

    print("Aplicación iniciada. Usa los siguientes comandos:")
    print("1. predict [ruta_imagen] - Para predecir el tamaño de una nueva imagen.")
    print("2. retrain - Para reentrenar el modelo con nuevos datos.")
    print("3. exit - Para salir de la aplicación.\n")

    while True:
        user_input = input("Introduce un comando: ").strip()

        if user_input.startswith("predict"):
            try:
                _, img_path = user_input.split(" ", 1)
                predicted_size = predict(model, label_encoder, img_path)
                print(f"Tamaño predicho: {predicted_size}")
            except FileNotFoundError as e:
                print(e)
            except Exception as e:
                print(f"Error al predecir: {e}")
        
        elif user_input == "retrain":
            try:
                print("Reentrenando el modelo...")
                data = load_data(json_file)
                features, labels = prepare_data(data)
                model, label_encoder = train_model(features, labels)
                joblib.dump((model, label_encoder), model_file)
                print("Modelo reentrenado y guardado.")
            except Exception as e:
                print(f"Error durante el reentrenamiento: {e}")

        elif user_input == "exit":
            print("Saliendo de la aplicación. ¡Hasta pronto!")
            break
        
        else:
            print("Comando no reconocido. Intenta nuevamente.")
