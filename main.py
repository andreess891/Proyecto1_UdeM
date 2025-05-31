from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import uvicorn

# Cargar el modelo
with open("modelo_clasificacion_ordenes.pkl", "rb") as f:
    modelo = pickle.load(f)

# Inicializar la app
app = FastAPI(title="API Clasificación de Órdenes EPM")

# Definir el esquema esperado para los datos de entrada
class OrdenInput(BaseModel):
    consumo_criticado: float
    estrato: float
    funcion_analisis: int
    periodicidad: int
    tipo_servicio: int
    codigo_observacion: int
    codigo_calificacion: int
    codigo_localidad: int
    codigo_categoria: int

@app.post("/predict")
def predecir_orden(data: OrdenInput):
    # Convertir a DataFrame
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])

    # TODO: Aplicar el mismo preprocesamiento que hiciste antes del entrenamiento:
    # - Encoding de variables categóricas
    # - Reordenar columnas si es necesario
    # - Imputación si había nulos, etc.

    # Este ejemplo asume que el modelo puede recibir directamente el DataFrame tal como está
    # En la práctica probablemente debas aplicar tu pipeline de transformación

    pred = modelo.predict(df_input)[0]
    return {"prediccion": int(pred), "significado": "1 = Acción, 0 = No acción"}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)
