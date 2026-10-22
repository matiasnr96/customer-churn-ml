
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import ClienteInput, PrediccionOutput
from src.inference.predict import cargar_modelo, predecir_cliente


# Modelo cargado al iniciar la API
modelo = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo

    modelo = cargar_modelo()
    yield
    modelo = None


app = FastAPI(
    title="Customer Churn API",
    description="API para predecir el abandono de clientes",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": modelo is not None
    }


@app.post("/predict", response_model=PrediccionOutput)
def predict(cliente: ClienteInput):

    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    # Convertir los datos validados a diccionario
    datos_cliente = cliente.model_dump()

    # Realizar predicción
    resultado = predecir_cliente(modelo, datos_cliente)

    return resultado
