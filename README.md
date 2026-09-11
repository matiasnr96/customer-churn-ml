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
