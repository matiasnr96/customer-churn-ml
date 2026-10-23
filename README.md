# Customer Churn ML

Proyecto integrador de la materia Laboratorio de Minería de Datos.

El objetivo del proyecto es predecir qué clientes tienen mayor probabilidad de abandonar el servicio utilizando técnicas de Machine Learning y buenas prácticas de MLOps.

## Objetivo

El problema consiste en predecir la variable `Churn`, que indica si un cliente abandonó o no el servicio.

Se trabajó con un dataset histórico de clientes y se realizaron las siguientes etapas:

- Análisis exploratorio de datos
- Tratamiento de valores faltantes
- Separación de datos de entrenamiento y prueba
- Preprocesamiento de variables numéricas y categóricas
- Entrenamiento de distintos modelos
- Comparación de métricas
- Registro de experimentos con MLflow
- Versionado del dataset con DVC
- Almacenamiento remoto de datos con DagsHub

## Estructura del proyecto

```text
customer-churn-ml/
|
|-- data/
|   `-- customer_churn_historical.csv
|
|-- notebooks/
|   `-- 01_eda.ipynb
|
|-- src/
|   `-- training/
|       `-- train.py
|
|-- models/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Dataset

El dataset histórico contiene 7043 clientes y 21 columnas.

La variable objetivo es `Churn`.

La columna `customerID` se utiliza únicamente como identificador y no se incluye como variable predictora.

También se detectaron valores faltantes en `TotalCharges`, los cuales se tratan dentro del pipeline de preprocesamiento.

## Modelos evaluados

Se realizaron distintos experimentos:

1. DummyClassifier como baseline
2. Regresión Logística
3. Random Forest
4. Regresión Logística con clases balanceadas
5. Random Forest con clases balanceadas
6. Regresión Logística con umbral de decisión 0.40

Las métricas utilizadas fueron:

- Accuracy
- Precision
- Recall
- F1-score

## Modelo seleccionado

Se seleccionó como modelo candidato una Regresión Logística con `class_weight="balanced"`.

Este modelo logró aproximadamente:

- Accuracy: 0.720
- Precision: 0.480
- Recall: 0.720
- F1-score: 0.576

Se priorizó el Recall porque uno de los principales objetivos del problema es detectar la mayor cantidad posible de clientes con riesgo de abandono.

## Instalación

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Recuperar los datos con DVC

El dataset se encuentra versionado mediante DVC y almacenado remotamente en DagsHub.

Para descargar los datos:

```bash
dvc pull
```

## Entrenamiento

Para entrenar el modelo:

```bash
python src/training/train.py
```

El script realiza:

- Carga del dataset
- Separación train/test
- Preprocesamiento
- Entrenamiento
- Evaluación
- Registro del experimento en MLflow

## MLflow

Los experimentos se registran con MLflow.

Para iniciar la interfaz local:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego ingresar desde el navegador a `http://127.0.0.1:5000`.

## Versionado

El código se versiona con Git y GitHub.

Los datos se versionan con DVC y se almacenan remotamente en DagsHub.


## Segundo parcial: API de inferencia y Docker

### Funcionalidades implementadas

El proyecto incorpora una API REST desarrollada con FastAPI para realizar predicciones de abandono de clientes (Customer Churn).

El modelo utilizado es una regresión logística entrenada con scikit-learn y registrada en MLflow. Para su ejecución en Docker, se utiliza una versión exportada del modelo y versionada mediante DVC.

La aplicación incluye:

- Endpoint `GET /health` para verificar el estado del servicio y la carga del modelo.
- Endpoint `POST /predict` para recibir los datos de un cliente y generar una predicción.
- Validación de datos mediante Pydantic.
- Documentación interactiva con Swagger.
- Tests automatizados con pytest.
- Contenerización con Docker y ejecución mediante Docker Compose.

### Requisitos

- Python 3.10.
- Docker Desktop con Docker Compose.
- Git y DVC.
- Acceso al almacenamiento remoto de DVC para recuperar los datos y el modelo.

### Instalación y ejecución

**1. Clonar el repositorio**

```bash
git clone https://github.com/matiasnr96/customer-churn-ml.git
cd customer-churn-ml
git checkout desarrollo-parcial-2
```

**2. Crear el entorno virtual e instalar dependencias**

En Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**3. Recuperar los archivos versionados con DVC**

```bash
dvc pull
```

Este comando recupera el dataset y el modelo exportado desde el almacenamiento remoto configurado en DVC. Puede requerir autenticación.

**4. Ejecutar la API con Docker Compose**

```bash
docker compose up --build -d
```

**5. Verificar el servicio**

- Estado de la API: http://localhost:8000/health
- Documentación Swagger: http://localhost:8000/docs

Desde Swagger se puede utilizar `POST /predict` para enviar los datos de un cliente y obtener una predicción.

La respuesta contiene:

- `prediction`: indica si el cliente abandonará el servicio (`Yes` o `No`).
- `probability`: probabilidad estimada de abandono.
- `risk_level`: clasificación de riesgo (`LOW`, `MEDIUM` o `HIGH`).
- `model_version`: versión del modelo utilizado.

**6. Ejecutar los tests**

En Windows PowerShell, con el entorno virtual activado:

```powershell
$env:MODEL_SOURCE = "local"
python -m pytest tests/ -v
```

En la última ejecución se aprobaron los 7 tests automatizados.

**7. Detener la aplicación**

```bash
docker compose down
```

### Versionado y reproducibilidad

Git se utiliza para versionar el código fuente y la configuración. DVC permite gestionar los archivos de datos y del modelo entrenado, almacenados de forma remota en DagsHub.

MLflow se utiliza para el seguimiento de experimentos y el registro del modelo. Para ejecutar la API dentro de Docker se utiliza el modelo exportado en `models/`, evitando depender de una instancia activa de MLflow.

El trabajo del segundo parcial se desarrolla en la rama `desarrollo-parcial-2`.