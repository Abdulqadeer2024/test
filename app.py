from flask import Flask, request, jsonify
from katabatic.models.production_model import ProductionModel
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ProductionModel()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        result = model.generate(data['data'])
        return jsonify({"prediction": result.tolist()})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    try:
        model.fit(data['X'], data['y'])
        return jsonify({"message": "Model trained successfully"})
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({"error": "Training failed"}), 500

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
