import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from model import predict_lulc
from utils import calculate_area_stats

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    output, classified, bounds = predict_lulc(filepath)
    stats = calculate_area_stats(classified)

    return jsonify({
        "output": output,
        "stats": stats,
        "bounds": [
            [bounds.bottom, bounds.left],
            [bounds.top, bounds.right]
        ]
    })

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
