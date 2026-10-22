import pandas as pd


def cargar_datos(ruta):
    df = pd.read_csv(ruta)

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    return X, y