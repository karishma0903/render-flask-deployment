from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os 

app = Flask(__name__)
CORS(app)

# Load PCA and models
with open("classification_pca.pkl", "rb") as f:
    classification_pca = pickle.load(f)

with open("classification_student_rf.pkl", "rb") as f:
    classification_model = pickle.load(f)

with open("recommendation_pca.pkl", "rb") as f:
    recommendation_pca = pickle.load(f)

with open("recommendation_student_rf.pkl", "rb") as f:
    recommendation_model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)
    features = np.array([
        data["PPBS"], data["GCT"], data["Height"], data["Weight of baby"],
        data["BP-DIASTOLE"], data["TSH"], data["FT4"]
    ]).reshape(1, -1)

    features_pca = classification_pca.transform(features)
    gdm_pred = classification_model.predict(features_pca)[0]
    gdm_pred_class =  int(round(gdm_pred))
    if gdm_pred_class == 0:
        gdm_type = "GDM Type I"
    elif gdm_pred_class == 1:
        gdm_type = "GDM Type II"
    else:
        gdm_type = "Unknown"

    rec_pca = recommendation_pca.transform(np.hstack((features, [[gdm_pred]])))
    recommendations_raw = recommendation_model.predict(rec_pca)[0]
    recommendation_labels = ["Exercise", "Walking", "Yoga", "Low Carb diet", "Dietary Fibres", "Life style changes", "Medication"]
    recommendations = [recommendation_labels[i] for i in range(len(recommendations_raw)) if recommendations_raw[i] >= 0.5]

    return jsonify({"gdm_type": gdm_type, "recommendations": recommendations})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Railway/Fly.io assigns the port automatically
    app.run(host='0.0.0.0', port=port)
    #app.run(host='127.0.0.1', port=5000, debug=True)
