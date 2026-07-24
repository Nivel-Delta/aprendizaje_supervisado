from pathlib import Path

from flask import Flask, jsonify, request
import pandas as pd
import pickle


BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "pipeline.pkl"

with PIPELINE_PATH.open("rb") as archivo:
    modelo = pickle.load(archivo)


app = Flask(__name__)

COLUMNAS_REQUERIDAS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]


@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Debes enviar un objeto JSON válido."
        }), 400

    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in data
    ]

    if faltantes:
        return jsonify({
            "error": "Faltan datos requeridos.",
            "columnas_faltantes": faltantes,
        }), 400

    try:
        input_data = pd.DataFrame([{
            columna: data[columna]
            for columna in COLUMNAS_REQUERIDAS
        }])

        prediccion = int(modelo.predict(input_data)[0])

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": f"Datos de entrada inválidos: {error}"
        }), 400

    resultado = (
        "sobrevivió"
        if prediccion == 1
        else "no sobrevivió"
    )

    return jsonify({
        "Survived": prediccion,
        "resultado": resultado,
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )