import joblib
import numpy as np

# Load model dictionary
model_data = joblib.load("knn_model.pkl")

knn_model = model_data["knn_model"]
pca = model_data["pca"]
scaler = model_data["scaler"]
label_to_id = model_data["label_to_id"]

# Buat reverse mapping: id → nama
id_to_label = {v: k for k, v in label_to_id.items()}


def predict_identity(feature_vector):
    """
    feature_vector: fitur hasil preprocessing (numpy array 1D)
    """

    # Pastikan bentuknya benar
    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Scaling
    scaled = scaler.transform(feature_vector)

    # PCA transform
    reduced = pca.transform(scaled)

    # Predict numeric ID
    pred_id = knn_model.predict(reduced)[0]

    # Convert ID → nama asli
    predicted_name = id_to_label.get(pred_id, "Unknown")

    return predicted_name
