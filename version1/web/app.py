# app.py

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# compile=False: avoids needing to register the custom focal_loss function
# at load time. We only use the model for inference, so this is safe.
model = load_model("../models/best_model.keras", compile=False)

with open("../models/class_names.json", "r") as f:
    CLASSES = json.load(f)

print("Loaded class order:", CLASSES)
# Expected: ['cercospora_leaf_spot', 'healthy', 'other_diseases']

# ── Load optimized thresholds saved by training script ───────────────────────
# The 2D grid search in mobilenetv2_train.py finds the cercospora and
# other_diseases thresholds that maximize macro F1 on the validation set.
# These are far more reliable than a hardcoded 0.90 whitelist, which was
# silently misclassifying any cercospora prediction below 90% as other_diseases
# and never directly predicting other_diseases at all.
_THRESHOLD_CONFIG_PATH = "../models/threshold_config.json"
if os.path.exists(_THRESHOLD_CONFIG_PATH):
    with open(_THRESHOLD_CONFIG_PATH) as f:
        _thresh = json.load(f)
    CERCOSPORA_THRESHOLD = _thresh.get("cercospora_threshold", 0.40)
    OD_THRESHOLD         = _thresh.get("od_threshold",         0.35)
    CERCOSPORA_IDX       = _thresh.get("cercospora_class_index", 0)
    OD_IDX               = _thresh.get("od_class_index",         2)
    HEALTHY_IDX          = _thresh.get("healthy_class_index",    1)
    print(f"Loaded thresholds — Cercospora: {CERCOSPORA_THRESHOLD}, OD: {OD_THRESHOLD}")
else:
    # Fallback defaults if threshold_config.json not yet generated.
    # Run mobilenetv2_train.py first to produce the optimized thresholds.
    CERCOSPORA_THRESHOLD = 0.40
    OD_THRESHOLD         = 0.35
    CERCOSPORA_IDX       = 0
    OD_IDX               = 2
    HEALTHY_IDX          = 1
    print("WARNING: threshold_config.json not found — using fallback defaults.")
    print("         Run mobilenetv2_train.py to generate optimized thresholds.")

TREATMENTS = {
    "cercospora_leaf_spot": (
        "CERCOSPORA LEAF SPOT detected! Remove infected leaves immediately, "
        "apply copper-based fungicide, and improve plant spacing for better airflow."
    ),
    "healthy": (
        "CHILI PLANT IS HEALTHY! "
        "Maintain proper watering, good air circulation, and regular fertilization."
    ),
    "other_diseases": (
        "OTHER DISEASE detected! Consult your local agricultural extension service "
        "for accurate diagnosis and treatment."
    ),
}

PLOTS_DIR = os.path.join("static", "plots")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_metrics():
    path = os.path.join(PLOTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def classify_with_thresholds(preds):
    """
    Apply training-optimized thresholds to raw model output probabilities.

    Priority order:
      1. If cercospora score >= CERCOSPORA_THRESHOLD → cercospora_leaf_spot
      2. Elif OD score >= OD_THRESHOLD               → other_diseases
      3. Else                                         → healthy

    WHY this order:
      - Cercospora causes the most visible crop damage and spreads fast.
        We want to catch it even at lower confidence, so it gets first priority.
      - other_diseases second — better to flag a possible disease than miss it.
      - healthy last — only predict healthy when neither disease threshold is met.

    WHY NOT the old CONFIDENCE_THRESHOLD=0.90 whitelist:
      - It silently mapped any cercospora prediction below 90% to other_diseases.
      - It made other_diseases unpredictable — it was the default fallback,
        not a genuine model prediction.
      - The 0.90 value had no empirical basis from the validation set.
      - These thresholds are derived from a 2D grid search maximizing macro F1.

    Returns:
      predicted_class (str), decision_score (float)
    """
    cercospora_score = float(preds[CERCOSPORA_IDX])
    od_score         = float(preds[OD_IDX])

    if cercospora_score >= CERCOSPORA_THRESHOLD:
        return "cercospora_leaf_spot", cercospora_score
    elif od_score >= OD_THRESHOLD:
        return "other_diseases", od_score
    else:
        healthy_score = float(preds[HEALTHY_IDX])
        return "healthy", healthy_score


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plots')
def plots():
    metrics = load_metrics()
    return render_template('plots.html', metrics=metrics)


@app.route('/api/metrics')
def api_metrics():
    metrics = load_metrics()
    if metrics is None:
        return jsonify({"error": "No metrics found. Run training first."}), 404
    return jsonify(metrics)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]

    # Threshold-based classification (replaces old 0.90 whitelist)
    predicted_class, decision_score = classify_with_thresholds(preds)

    # Build full probability dict keyed by class name
    probs_dict = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

    # Chart data in fixed display order for frontend consistency
    chart_classes = ["cercospora_leaf_spot", "healthy", "other_diseases"]
    chart_probs   = [
        probs_dict.get("cercospora_leaf_spot", 0.0),
        probs_dict.get("healthy",              0.0),
        probs_dict.get("other_diseases",       0.0),
    ]

    # Raw argmax for logging / transparency
    raw_idx   = int(np.argmax(preds))
    raw_class = CLASSES[raw_idx]
    raw_conf  = float(preds[raw_idx])

    treatment = TREATMENTS.get(predicted_class, "Consult an agricultural expert.")

    print(
        f"RAW argmax: {raw_class} ({raw_conf:.1%}) | "
        f"Cercospora={preds[CERCOSPORA_IDX]:.1%} (thresh={CERCOSPORA_THRESHOLD}) | "
        f"OD={preds[OD_IDX]:.1%} (thresh={OD_THRESHOLD}) | "
        f"→ FINAL: {predicted_class}"
    )

    all_probs = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}

    return jsonify({
        "prediction":     predicted_class,
        "confidence":     round(decision_score * 100, 2),
        "raw_class":      raw_class,
        "raw_confidence": round(raw_conf * 100, 2),
        "treatment":      treatment,
        "image_url":      f"/{filepath.replace(os.sep, '/')}",
        "probabilities":  chart_probs,
        "classes":        chart_classes,
        "all_probs":      all_probs,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    