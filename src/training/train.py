import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Ruta del dataset
DATA_PATH = "data/customer_churn_historical.csv"


# Cargar datos
df = pd.read_csv(DATA_PATH)


# Separar variables predictoras y objetivo
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]


# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Columnas numéricas
columnas_numericas = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]


# Columnas categóricas
columnas_categoricas = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# Pipeline para variables numéricas
pipeline_numerico = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler())
])


# Pipeline para variables categóricas
pipeline_categorico = Pipeline([
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# Unir ambos tipos de preprocesamiento
preprocesador = ColumnTransformer([
    ("numericas", pipeline_numerico, columnas_numericas),
    ("categoricas", pipeline_categorico, columnas_categoricas)
])


# Modelo candidato
modelo = Pipeline([
    ("preprocesamiento", preprocesador),
    ("modelo", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


# Entrenar modelo
modelo.fit(X_train, y_train)


print("Modelo entrenado correctamente")
print("Datos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Yes")
recall = recall_score(y_test, y_pred, pos_label="Yes")
f1 = f1_score(y_test, y_pred, pos_label="Yes")

print("\nResultados del modelo")
print("---------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)


# Configuración de MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer-churn-experimentos")


# Registrar experimento
with mlflow.start_run(run_name="train_logistica_balanceada"):

    # Parámetros
    mlflow.log_param("modelo", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # Modelo
    mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        serialization_format="cloudpickle"
    )


print("\nExperimento registrado correctamente en MLflow")