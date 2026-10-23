
import pandas as pd
import pytest

from src.inference.predict import cargar_modelo, predecir_cliente


@pytest.fixture(scope="module")
def modelo():
    # Cargar el modelo una sola vez
    return cargar_modelo()


@pytest.fixture(scope="module")
def datos_cliente():
    # Utilizar un cliente real del dataset
    df = pd.read_csv("data/customer_churn_historical.csv")

    return df.drop(
        columns=["Churn", "customerID"]
    ).iloc[0].to_dict()


def test_modelo_cargado(modelo):
    assert modelo is not None
    assert hasattr(modelo, "predict")
    assert hasattr(modelo, "predict_proba")


def test_prediccion_valida(modelo, datos_cliente):
    resultado = predecir_cliente(modelo, datos_cliente)

    assert resultado["prediction"] in ["Yes", "No"]
    assert 0 <= resultado["probability"] <= 1
    assert resultado["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert resultado["model_version"] == "1"


def test_clasificacion_riesgo(modelo, datos_cliente):
    resultado = predecir_cliente(modelo, datos_cliente)

    probabilidad = resultado["probability"]
    riesgo = resultado["risk_level"]

    if probabilidad < 0.30:
        assert riesgo == "LOW"
    elif probabilidad < 0.60:
        assert riesgo == "MEDIUM"
    else:
        assert riesgo == "HIGH"
