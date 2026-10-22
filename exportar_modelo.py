
from pathlib import Path

import mlflow
import mlflow.artifacts

# Configurar MLflow local
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Modelo registrado
MODELO_URI = "models:/customer-churn-model@candidate"

# Carpeta donde exportaremos el modelo
DESTINO = Path("models")
DESTINO.mkdir(exist_ok=True)

# Descargar el modelo registrado
ruta = mlflow.artifacts.download_artifacts(
    artifact_uri=MODELO_URI,
    dst_path=str(DESTINO)
)

print("Modelo exportado correctamente")
print("Ubicación:", ruta)
