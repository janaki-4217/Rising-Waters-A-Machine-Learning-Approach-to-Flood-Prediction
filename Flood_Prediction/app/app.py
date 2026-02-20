from flask import Flask, render_template, request
import numpy as np
import os
import pickle

# Create Flask app
app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
model_path = os.path.join(BASE_DIR, "..", "models", "logreg_pipeline.pkl")

# Load model safely
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['temp']),
                float(request.form['humidity']),
                float(request.form['cloud_cover']),
                float(request.form['annual_rainfall']),
                float(request.form['rain_janfeb']),
                float(request.form['rain_mar_may']),
                float(request.form['rain_jun_sep']),
                float(request.form['rain_octdec']),
                float(request.form['avg_june']),
                float(request.form['sub'])
            ]

            final_input = np.array([features])

            probability = model.predict_proba(final_input)[0][1]
            percent = round(probability * 100, 2)

            if percent <= 50:
                return render_template('low.html', percent=percent)
            elif percent <= 75:
                return render_template('mid.html', percent=percent)
            else:
                return render_template('high.html', percent=percent)

        except Exception as e:
            return f"Error: {e}"

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)