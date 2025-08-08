from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("models/horn_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.expand_dims(mfcc, axis=0)

    pred = model.predict(mfcc)[0][0]
    return jsonify({"horn_probability": float(pred)})

@app.route("/", methods=["GET"])
def home():
    return "Horn detection API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
