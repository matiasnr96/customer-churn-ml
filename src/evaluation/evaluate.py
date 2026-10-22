from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluar_modelo(modelo, X_test, y_test):

    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Calcular métricas
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Yes"),
        "recall": recall_score(y_test, y_pred, pos_label="Yes"),
        "f1": f1_score(y_test, y_pred, pos_label="Yes")
    }

    return metricas