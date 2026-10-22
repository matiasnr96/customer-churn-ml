
import sys
from pathlib import Path

import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Permitir importar los módulos desde src
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.data.load_data import cargar_datos
from src.preprocessing.preprocess import crear_preprocesador
from src.evaluation.evaluate import evaluar_modelo


def main():

    # Leer configuración
    with open(RAIZ / "config.yaml", "r", encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo)

    # Cargar datos
    X, y = cargar_datos(RAIZ / config["data"]["path"])

    # Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    # Crear el modelo con su preprocesamiento
    modelo = Pipeline([
        ("preprocesamiento", crear_preprocesador()),
        ("modelo", LogisticRegression(
            max_iter=config["model"]["max_iter"],
            class_weight=config["model"]["class_weight"]
        ))
    ])

    # Entrenar
    modelo.fit(X_train, y_train)

    print("Modelo entrenado correctamente")
    print("Datos de entrenamiento:", X_train.shape)
    print("Datos de prueba:", X_test.shape)

    # Evaluar
    metricas = evaluar_modelo(modelo, X_test, y_test)

    print("\nResultados del modelo")
    print("---------------------")

    for nombre, valor in metricas.items():
        print(f"{nombre.capitalize()}: {valor}")

    # Configurar MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Registrar experimento
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):

        mlflow.log_param("modelo", config["model"]["type"])
        mlflow.log_param("max_iter", config["model"]["max_iter"])
        mlflow.log_param("class_weight", config["model"]["class_weight"])
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_state", config["data"]["random_state"])

        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            serialization_format="cloudpickle"
        )

    print("\nExperimento registrado correctamente en MLflow")


if __name__ == "__main__":
    main()
