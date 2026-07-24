from pathlib import Path
from flask import Flask, request, jsonify
import pickle
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo.pkl"

with MODEL_PATH.open("rb") as archivo:
    modelo = pickle.load(archivo)

app = Flask(__name__)


@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json(silent=True)

    if not data or "input" not in data:
        return jsonify({
            "error": "Debes enviar un JSON con la clave input."
        }), 400

    try:
        input_data = np.asarray(
            data["input"],
            dtype=float
        ).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({
            "error": "Los valores de input deben ser numéricos."
        }), 400

    if input_data.shape[1] != 7:
        return jsonify({
            "error": "Se requieren exactamente 7 valores."
        }), 400

    prediccion = int(modelo.predict(input_data)[0])

    return jsonify({
        "prediccion": prediccion
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )