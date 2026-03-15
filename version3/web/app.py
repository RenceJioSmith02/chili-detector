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

model = load_model("../models/best_model.keras", compile=False)

with open("../models/class_names.json", "r") as f:
    CLASSES = json.load(f)

print("Loaded class order:", CLASSES)
# Expected: ['cercospora_leaf_spot', 'healthy', 'other_diseases']

# ── Load optimized thresholds saved by training script ───────────────────────
# Priority order:
#   1. cercospora_leaf_spot  — highest crop damage risk, catch it first
#   2. healthy               — preferred over catch-all when confident
#   3. other_diseases        — last resort, only when neither threshold is met
_THRESHOLD_CONFIG_PATH = "../models/threshold_config.json"
if os.path.exists(_THRESHOLD_CONFIG_PATH):
    with open(_THRESHOLD_CONFIG_PATH) as f:
        _thresh = json.load(f)
    CERCOSPORA_THRESHOLD = _thresh.get("cercospora_threshold", 0.45)
    HEALTHY_THRESHOLD    = _thresh.get("healthy_threshold",    0.45)
    CERCOSPORA_IDX       = _thresh.get("cercospora_class_index", 0)
    OD_IDX               = _thresh.get("od_class_index",         2)
    HEALTHY_IDX          = _thresh.get("healthy_class_index",    1)
    print(f"Loaded thresholds — Cercospora: {CERCOSPORA_THRESHOLD}, Healthy: {HEALTHY_THRESHOLD}")
else:
    # Fallback defaults if threshold_config.json not yet generated.
    # Run mobilenetv2_train.py first to produce the optimized thresholds.
    CERCOSPORA_THRESHOLD = 0.45
    HEALTHY_THRESHOLD    = 0.45
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
      2. Elif healthy score  >= HEALTHY_THRESHOLD    → healthy
      3. Else                                         → other_diseases  (last resort)
    """
    cercospora_score = float(preds[CERCOSPORA_IDX])
    healthy_score    = float(preds[HEALTHY_IDX])

    if cercospora_score >= CERCOSPORA_THRESHOLD:
        return "cercospora_leaf_spot", cercospora_score
    elif healthy_score >= HEALTHY_THRESHOLD:
        return "healthy", healthy_score
    else:
        od_score = float(preds[OD_IDX])
        return "other_diseases", od_score   # last resort


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

    # Threshold-based classification (priority: cercospora → healthy → other_diseases)
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
        f"Healthy={preds[HEALTHY_IDX]:.1%} (thresh={HEALTHY_THRESHOLD}) | "
        f"OD={preds[OD_IDX]:.1%} (last resort) | "
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