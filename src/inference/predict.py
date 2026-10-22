
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


# Carpeta principal del proyecto
RAIZ = Path(__file__).resolve().parents[2]

# Configurar MLflow local
mlflow.set_tracking_uri(
    f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
)

# Modelo registrado en MLflow
MODELO_REGISTRADO = "models:/customer-churn-model@candidate"

# Modelo exportado para Docker
MODELO_LOCAL = RAIZ / "models"


def cargar_modelo():
    # Elegir desde dónde cargar el modelo
    origen = os.getenv("MODEL_SOURCE", "registry")

    if origen == "local":
        print("Cargando modelo exportado desde models/")
        modelo = mlflow.sklearn.load_model(
            str(MODELO_LOCAL)
        )

    elif origen == "registry":
        print("Cargando modelo desde MLflow Registry")
        modelo = mlflow.sklearn.load_model(
            MODELO_REGISTRADO
        )

    else:
        raise ValueError(
            "MODEL_SOURCE debe ser 'local' o 'registry'"
        )

    return modelo


def predecir_cliente(modelo, datos_cliente):
    # Convertir los datos del cliente a DataFrame
    df = pd.DataFrame([datos_cliente])

    # Obtener predicción
    prediccion = modelo.predict(df)[0]

    # Calcular probabilidad de abandono
    clases = list(modelo.classes_)
    indice_churn = clases.index("Yes")
    probabilidad = modelo.predict_proba(df)[0][indice_churn]

    # Clasificar riesgo
    if probabilidad < 0.30:
        riesgo = "LOW"
    elif probabilidad < 0.60:
        riesgo = "MEDIUM"
    else:
        riesgo = "HIGH"

    return {
        "prediction": str(prediccion),
        "churn_probability": round(float(probabilidad), 4),
        "risk_level": riesgo
    }
