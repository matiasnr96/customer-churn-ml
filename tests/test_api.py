
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


# Cliente de prueba que inicia la API y carga el modelo
def crear_cliente():
    return TestClient(app)


# Obtener un cliente válido del dataset
def obtener_datos_cliente():
    df = pd.read_csv("data/customer_churn_historical.csv")

    datos = df.drop(
        columns=["Churn", "customerID"]
    ).iloc[0].to_dict()

    # Convertir valores numéricos al formato JSON
    return datos


def test_health():
    with crear_cliente() as cliente:
        respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"
    assert respuesta.json()["model_loaded"] is True


def test_predict_valido():
    datos = obtener_datos_cliente()

    with crear_cliente() as cliente:
        respuesta = cliente.post("/predict", json=datos)

    assert respuesta.status_code == 200

    resultado = respuesta.json()

    assert resultado["prediction"] in ["Yes", "No"]
    assert 0 <= resultado["probability"] <= 1
    assert resultado["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert resultado["model_version"] == "1"


def test_predict_tenure_negativo():
    datos = obtener_datos_cliente()
    datos["tenure"] = -5

    with crear_cliente() as cliente:
        respuesta = cliente.post("/predict", json=datos)

    assert respuesta.status_code == 422


def test_predict_campo_faltante():
    datos = obtener_datos_cliente()
    del datos["gender"]

    with crear_cliente() as cliente:
        respuesta = cliente.post("/predict", json=datos)

    assert respuesta.status_code == 422
