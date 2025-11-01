import streamlit as st
import requests
import json
import pandas as pd
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Configuración de la página
st.set_page_config(
    page_title="Predicción órdenes de desvío",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de la API
load_dotenv()
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

def make_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza una predicción usando el endpoint de Databricks
    """
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "instances": [data]
    }
    
    try:
        response = requests.post(
            DATABRICKS_URL, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"Error {response.status_code}: {response.text}",
                "status_code": response.status_code
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Error de conexión: {str(e)}",
            "status_code": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error interno: {str(e)}",
            "status_code": None
        }

def load_sample_data() -> Dict[str, Any]:
    """
    Carga datos de ejemplo para la predicción
    """
    return {
        "consumo_criticado": 150.5,
        "categoria": "1-RESIDENCIAL",
        "nivel_tension": 220,
        "estrato": 3,
        "localidad": "5001-MEDELLÍN",
        "funcion_analisis": "CALCCOLE - Calcular Consumo por Lecturas",
        "calificacion": "5019-ALTO ENERGIA (>100%)",
        "obs_lectura": "21-FRAUDE",
        "periodicidad": 1,
        "tipo_servicio": "Energía"
    }

def main():
    """
    Función principal de la aplicación Streamlit
    """
    
    # Título principal
    st.title("Predicción órdenes de desvío")
    st.markdown("---")
    
    # Descripción
    st.markdown("""
    Esta aplicación utiliza un modelo de machine learning hospedado en Databricks 
    para predecir el desvío de consumos basado en diferentes características del servicio.
    """)
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("Configuración")
        
        # Opción para cargar datos de ejemplo
        if st.button("Cargar Datos de Ejemplo", type="secondary"):
            sample_data = load_sample_data()
            for key, value in sample_data.items():
                st.session_state[key] = value
            st.success("Datos de ejemplo cargados!")
            st.rerun()
        
        # Información de la API
        st.markdown("### Información de API")
        st.text("Endpoint: Databricks ML")
    
    # Formulario principal
    st.header("Datos para Predicción")
    
    # Crear columnas para organizar mejor el formulario
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Información Básica")
        
        consumo_criticado = st.number_input(
            "Consumo Criticado",
            min_value=0.0,
            value=st.session_state.get("consumo_criticado", 0.0),
            step=0.1,
            help="Valor del consumo criticado en kWh"
        )
        
        categoria_options = [
            "1-RESIDENCIAL"
            ,"2-COMERCIAL"
            ,"3-INDUSTRIAL"
            ,"4-OFICIAL"
            ,"5-ESPECIAL"
            ]
        categoria_value = st.session_state.get("categoria", 0)
        # Si el valor guardado es un string, convertirlo al índice correspondiente
        if isinstance(categoria_value, str) and categoria_value in categoria_options:
            categoria_index = categoria_options.index(categoria_value)
        elif isinstance(categoria_value, int):
            categoria_index = categoria_value
        else:
            categoria_index = 0
        
        categoria = st.selectbox(
            "Categoría",
            options=categoria_options,
            index=categoria_index,
            help="Tipo de categoría del servicio (ej: Residencial, Comercial)"
        )
        
        nivel_tension = st.number_input(
            "Nivel de Tensión",
            min_value=0,
            value=st.session_state.get("nivel_tension", 0),
            step=1,
            help="Nivel de tensión del servicio"
        )
        
        estrato = st.selectbox(
            "Estrato",
            options=[1, 2, 3, 4, 5, 6],
            index=st.session_state.get("estrato", 1) - 1 if "estrato" in st.session_state else 0,
            help="Estrato socioeconómico (1-6)"
        )
        
        localidad_options = [
            "5001-MEDELLÍN"
            ,"5088-BELLO"
            ,"5360-ITAGUI"
            ,"5266-ENVIGADO"
            ,"5308-GIRARDOTA"
            ,"5615-RIONEGRO"
            ,"5490-NECOCLÍ"
            ]
        localidad_value = st.session_state.get("localidad", 0)
        # Si el valor guardado es un string, convertirlo al índice correspondiente
        if isinstance(localidad_value, str) and localidad_value in localidad_options:
            localidad_index = localidad_options.index(localidad_value)
        elif isinstance(localidad_value, int):
            localidad_index = localidad_value
        else:
            localidad_index = 0
            
        localidad = st.selectbox(
            "Localidad",
            options=localidad_options,
            index=localidad_index,
            help="Localidad o zona geográfica"
        )
    
    with col2:
        st.subheader("Información Técnica")
        
        funcion_analisis_options = [
            "CALCCOPR - Calcular Consumo Penalizado de Energía Reactiva"
            ,"CALCCOLE - Calcular Consumo por Lecturas"
            ,"PEOBTECOPP - Obtener Consumo Promedio Individual últimos meses parametrizados"
            ,"PECALCCOLA - Calcular Consumo por Lectura Anterior"
            ,"PESTACONS - Establecer Consumo Manual"
            ,"PEDISCONEQ - Distribución Equitativa de los Consumos de un Período"
            ,"PEOBTECOPS - Obtener Consumo Promedio Subcategoría"
        ]
        funcion_analisis_value = st.session_state.get("funcion_analisis", 0)
        # Si el valor guardado es un string, convertirlo al índice correspondiente
        if isinstance(funcion_analisis_value, str) and funcion_analisis_value in funcion_analisis_options:
            funcion_analisis_index = funcion_analisis_options.index(funcion_analisis_value)
        elif isinstance(funcion_analisis_value, int):
            funcion_analisis_index = funcion_analisis_value
        else:
            funcion_analisis_index = 0
        
        funcion_analisis = st.selectbox(
            "Función Análisis",
            options=funcion_analisis_options,
            index=funcion_analisis_index,
            help="Función o tipo de análisis aplicado"
        )
        
        calificacion_options = [
            "2-ENVÍA A CRÍTICA"
            ,"48-POSIBLE INVESTIGACION"
            ,"5007-NORMAL ENERGIA(-80% A 80%)"
            ,"5035-BAJO ENERGIA (<-50%)"
            ,"5019-ALTO ENERGIA (>100%)"
            ,"5080-MUY ALTO (>500%)"
            ,"5085-ESTIMACION VALIDA"
            ,"220-CONSUMO POR ESTIMACIONES MAYORES A PARAMETRO"
        ]
        calificacion_value = st.session_state.get("calificacion", 0)
        # Si el valor guardado es un string, convertirlo al índice correspondiente
        if isinstance(calificacion_value, str) and calificacion_value in calificacion_options:
            calificacion_index = calificacion_options.index(calificacion_value)
        elif isinstance(calificacion_value, int):
            calificacion_index = calificacion_value
        else:
            calificacion_index = 0
        
        calificacion = st.selectbox(
            "Calificación",
            options=calificacion_options,
            index=calificacion_index,
            help="Calificación sobre la lectura"
        )
        
        obs_lectura_options = [
            "0-SIN CAUSA NI OBSERVACIÓN"
            ,"1-NO EXISTE GEOGRAFICAMENTE"
            ,"2-IMPOSIBILIDAD DE ACCESO"
            ,"3-PROFUNDO O MUY ALTO"
            ,"7-MEDIDOR DIGITAL"
            ,"19-REPARÓ DAÑO O FUGA"
            ,"21-FRAUDE"
            ,"53-EXCLUIDO POR PERIODICIDAD"
        ]
        obs_lectura_value = st.session_state.get("obs_lectura", 0)
        # Si el valor guardado es un string, convertirlo al índice correspondiente
        if isinstance(obs_lectura_value, str) and obs_lectura_value in obs_lectura_options:
            obs_lectura_index = obs_lectura_options.index(obs_lectura_value)
        elif isinstance(obs_lectura_value, int):
            obs_lectura_index = obs_lectura_value
        else:
            obs_lectura_index = 0
        
        obs_lectura = st.selectbox(
            "Observación Lectura",
            options=obs_lectura_options,
            index=obs_lectura_index,
            help="Observaciones sobre la lectura"
        )
        
        periodicidad_value = st.session_state.get("periodicidad", 1)
        # Asegurar que periodicidad esté en el rango válido (1-3)
        if isinstance(periodicidad_value, int) and 1 <= periodicidad_value <= 3:
            periodicidad_index = periodicidad_value - 1
        else:
            periodicidad_index = 0
            
        periodicidad = st.selectbox(
            "Periodicidad",
            options=[1, 2, 3],
            index=periodicidad_index,
            help="Periodicidad (1-Mensual, 2-Bimestral, 3-Trimestral)"
        )
        
        tipo_servicio_options = ["Energía", "Agua", "Gas", "Alcantarillado"]
        tipo_servicio_value = st.session_state.get("tipo_servicio", 0)
        # Si el valor guardado es un string, convertirlo al índice correspondiente
        if isinstance(tipo_servicio_value, str) and tipo_servicio_value in tipo_servicio_options:
            tipo_servicio_index = tipo_servicio_options.index(tipo_servicio_value)
        elif isinstance(tipo_servicio_value, int):
            tipo_servicio_index = tipo_servicio_value
        else:
            tipo_servicio_index = 0
        
        tipo_servicio = st.selectbox(
            "Tipo de Servicio",
            options=tipo_servicio_options,
            index=tipo_servicio_index,
            help="Tipo de servicio en revisión"
        )
    
    # Botón para realizar predicción
    st.markdown("---")
    
    if st.button("Obtener Predicción", type="primary", use_container_width=True):
        
        # Validar que todos los campos estén llenos
        required_fields = {
            "consumo_criticado": consumo_criticado,
            "categoria": categoria,
            "nivel_tension": nivel_tension,
            "estrato": estrato,
            "localidad": localidad,
            "funcion_analisis": funcion_analisis,
            "calificacion": calificacion,
            "obs_lectura": obs_lectura,
            "periodicidad": periodicidad,
            "tipo_servicio": tipo_servicio
        }
        
        empty_fields = [field for field, value in required_fields.items() 
                       if value == "" or value is None]
        
        if empty_fields:
            st.error(f"Por favor completa los siguientes campos: {', '.join(empty_fields)}")
        else:
            # Mostrar spinner mientras se procesa
            with st.spinner("Procesando predicción..."):
                
                # Preparar datos para la API
                prediction_data = {
                    "consumo_criticado": float(consumo_criticado),
                    "categoria": categoria,
                    "nivel_tension": int(nivel_tension),
                    "estrato": int(estrato),
                    "localidad": localidad,
                    "funcion_analisis": funcion_analisis,
                    "calificacion": calificacion,
                    "obs_lectura": obs_lectura,
                    "periodicidad": int(periodicidad),
                    "tipo_servicio": tipo_servicio
                }
                
                # Realizar predicción
                result = make_prediction(prediction_data)
                
                # Mostrar resultados
                if result["success"]:
                    st.success("¡Predicción realizada exitosamente!")
                    
                    # Mostrar datos enviados
                    with st.expander("Datos Enviados"):
                        st.json(prediction_data)
                    
                    # Mostrar respuesta
                    st.subheader("Resultado de la Predicción")
                    
                    # Crear un contenedor para la respuesta
                    response_data = result["data"]
                    
                    # Mostrar la respuesta en formato JSON más legible
                    st.json(response_data)
                    
                    # Si hay predicciones específicas, mostrarlas de forma más clara
                    if "predictions" in response_data:
                        st.subheader("Predicciones")
                        predictions = response_data["predictions"]
                        
                        if isinstance(predictions, list) and predictions:
                            for i, pred in enumerate(predictions):
                                st.metric(f"Predicción {i+1}", f"{pred}")
                
                else:
                    st.error("Error al realizar la predicción")
                    st.error(result["error"])
                    
                    # Mostrar información adicional para debugging
                    with st.expander("Información de Debug"):
                        st.write("**Status Code:**", result.get("status_code", "N/A"))
                        st.write("**Datos enviados:**")
                        st.json(prediction_data)
                        
                        st.markdown("""
                        **Verifica lo siguiente:**
                        1. El endpoint de Databricks está disponible
                        2. El token de autorización es válido
                        3. Los datos están en el formato correcto
                        4. La conexión a internet está activa
                        """)

    # Información adicional en la parte inferior
    st.markdown("---")
    st.markdown("""
    ### Información del Modelo
    - **Plataforma**: Databricks ML
    - **Propósito**: Predicción órdenes de desvío
    """)

if __name__ == "__main__":
    main()