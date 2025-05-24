from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

# Cargar el modelo
with open("modelo_clasificacion_ordenes.pkl", "rb") as f:
    modelo = pickle.load(f)

# Inicializar la app
app = FastAPI(title="API Clasificación de Órdenes EPM")

# Definir el esquema esperado para los datos de entrada
class OrdenInput(BaseModel):
    CONSUMO_CRITICADO: float
    SERVICIO: str
    CATEGORIA: str
    NIVEL_TENSION: str
    ESTRATO: int
    LOCALIDAD: str
    FUNCION_ANALISIS: str
    CALIFICACION: str
    OBS_LECTURA: str
    PERIODICIDAD: int

@app.post("/predict")
def predecir_orden(data: OrdenInput):
    # Convertir a DataFrame
    input_dict = data.dic()
    df_input = pd.DataFrame([input_dict])

    # TODO: Aplicar el mismo preprocesamiento que hiciste antes del entrenamiento:
    # - Encoding de variables categóricas
    # - Reordenar columnas si es necesario
    # - Imputación si había nulos, etc.

    # Este ejemplo asume que el modelo puede recibir directamente el DataFrame tal como está
    # En la práctica probablemente debas aplicar tu pipeline de transformación

    pred = modelo.predict(df_input)[0]
    return {"prediccion": int(pred), "significado": "1 = Acción, 0 = No acción"}
