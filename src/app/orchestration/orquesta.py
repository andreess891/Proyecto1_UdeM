import os
import sys
import logging
import importlib.util
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_table_artifact
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import mlflow

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agregar el directorio actual al path para importaciones
current_dir = Path(__file__).parent  # orchestration/
app_dir = current_dir.parent         # app/
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(app_dir.parent))  # src/

# Importar las clases del proyecto
def import_project_classes():
    """Importa las clases del proyecto manejando diferentes escenarios."""
    try:
        # Primero intentar importación directa
        from data.etl import GetData
        from data.feature_engineering import FeatureEngineering
        from train.train import TrainModel
        return GetData, FeatureEngineering, TrainModel
    except ImportError:
        try:
            # Intentar con path absoluto - el script está en orchestration/, necesitamos ir a app/
            app_dir = current_dir.parent  # Subir de orchestration/ a app/
            
            # Importar GetData
            etl_path = app_dir / "data" / "etl.py"
            spec = importlib.util.spec_from_file_location("data.etl", etl_path)
            etl_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(etl_module)
            GetData = etl_module.GetData
            
            # Importar FeatureEngineering
            fe_path = app_dir / "data" / "feature_engineering.py"
            spec = importlib.util.spec_from_file_location("data.feature_engineering", fe_path)
            fe_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fe_module)
            FeatureEngineering = fe_module.FeatureEngineering
            
            # Importar TrainModel
            train_path = app_dir / "train" / "train.py"
            spec = importlib.util.spec_from_file_location("train.train", train_path)
            train_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_module)
            TrainModel = train_module.TrainModel
            
            return GetData, FeatureEngineering, TrainModel
            
        except Exception as e:
            app_dir = current_dir.parent  # Definir app_dir para los mensajes de error
            logger.info(f"Error importando módulos del proyecto: {e}")
            logger.info("Verifica que los archivos existen en las siguientes rutas:")
            logger.info(f"   - {app_dir / 'data' / 'etl.py'}")
            logger.info(f"   - {app_dir / 'data' / 'feature_engineering.py'}")
            logger.info(f"   - {app_dir / 'train' / 'train.py'}")
            raise ImportError("No se pudieron importar los módulos del proyecto")

# Importar las clases
try:
    GetData, FeatureEngineering, TrainModel = import_project_classes()
    logger.info("Módulos del proyecto importados exitosamente")
except ImportError as e:
    logger.info(f"Error crítico: {e}")
    sys.exit(1)

# Función para resolver rutas de archivos
def resolve_data_path(file_path: str) -> str:
    if os.path.isabs(file_path):
        if os.path.exists(file_path):
            return file_path
    
    # El script está en src/app/orchestration/, necesitamos ir hacia atrás para llegar a src/app/
    app_dir = current_dir.parent  # Subir de orchestration/ a app/
    
    # Buscar el archivo en diferentes ubicaciones posibles
    possible_paths = [
        # 1. Desde app_dir con la ruta relativa proporcionada
        app_dir / file_path,
        # 2. Ruta específica para BD_ordenes.xlsx en src/app/data/raw_data/
        app_dir / "data" / "raw_data" / "BD_ordenes.xlsx",
        # 3. Desde current_dir (orchestration/) con la ruta proporcionada
        current_dir / file_path,
        # 4. Ruta directa con nombre de archivo solamente
        app_dir / "data" / "raw_data" / Path(file_path).name,
        # 5. Desde raíz del proyecto (subir tres niveles)
        current_dir.parent.parent.parent / file_path,
        # 6. Buscar en directorio actual de ejecución
        Path.cwd() / file_path,
        # 7. Buscar en directorio actual de ejecución + estructura
        Path.cwd() / "src" / "app" / "data" / "raw_data" / "BD_ordenes.xlsx"
    ]
    
    logger.info(f"Buscando archivo: {file_path}")
    
    for i, path in enumerate(possible_paths, 1):
        logger.info(f"  {i}. Verificando: {path}")
        if path.exists():
            logger.info(f"Archivo encontrado en: {path}")
            return str(path.absolute())
        else:
            logger.info(f"No existe: {path}")
    
    # Si no encuentra el archivo, mostrar información de debug
    logger.error(f"Archivo no encontrado: {file_path}")
    logger.error(f"Directorio actual del script: {current_dir}")
    logger.error(f"Directorio de trabajo: {Path.cwd()}")
    logger.error(f"Se buscó en {len(possible_paths)} ubicaciones")
    
    # Retornar la ruta más probable para que el error sea claro
    return str(app_dir / "data" / "raw_data" / "BD_ordenes.xlsx")

# MLflow configuration with fallback
def setup_mlflow():
    """Setup MLflow with proper error handling and fallback options."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Test connection
        mlflow.search_experiments()
        logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {mlflow_uri}: {e}")
        logger.info("Falling back to local SQLite database")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        mlflow.set_experiment("Energy-Respuesta-experiment-prefect")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise

# Initialize MLflow
setup_mlflow()


@task(name="extract_and_load_data", retries=2, retry_delay_seconds=10)
def extract_and_load_data(file_path: str = "data/raw_data/BD_ordenes.xlsx") -> pd.DataFrame:
    task_logger = get_run_logger()
    task_logger.info(f"Iniciando extracción de datos desde: {file_path}")
    
    try:
        # Resolver la ruta completa del archivo
        resolved_path = resolve_data_path(file_path)
        task_logger.info(f"Ruta resuelta: {resolved_path}")
        
        # Verificar que el archivo existe
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"El archivo no existe en la ruta: {resolved_path}")
        
        # Instanciar la clase GetData
        data_extractor = GetData(file_path=resolved_path)
        
        # Leer los datos
        df = data_extractor.read_data()
        
        task_logger.info(f"Datos extraídos exitosamente. Shape: {df.shape}")
        task_logger.info(f"Columnas: {list(df.columns)}")
        
        # Crear artefacto de tabla con resumen de los datos
        data_summary = [
            ["Número de filas", int(df.shape[0])],
            ["Número de columnas", int(df.shape[1])],
            ["Archivo fuente", str(resolved_path)],
            ["Columnas", ", ".join(df.columns.tolist())],
            ["Valores nulos totales", int(df.isnull().sum().sum())]
        ]
        
        create_table_artifact(
            key="data-extraction-summary",
            table=data_summary,
            description="Resumen de extracción de datos"
        )
        
        return df
        
    except Exception as e:
        task_logger.error(f"Error al extraer datos: {str(e)}")
        raise


@task(name="feature_engineering", retries=1, retry_delay_seconds=5)
def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    task_logger = get_run_logger()
    task_logger.info("Iniciando ingeniería de características")
    
    try:
        # Instanciar la clase FeatureEngineering
        feature_engineer = FeatureEngineering(df)
        
        # Preparar características específicas para energía
        df_processed, target_variable, numeric_features, categorical_features = feature_engineer.prepare_features_only_energy()
        
        task_logger.info(f"Ingeniería de características completada. Shape final: {df_processed.shape}")
        task_logger.info(f"Variable objetivo: {target_variable}")
        task_logger.info(f"Características numéricas ({len(numeric_features)}): {numeric_features}")
        task_logger.info(f"Características categóricas ({len(categorical_features)}): {categorical_features}")
        
        # Crear artefacto de tabla con resumen de feature engineering
        feature_summary = [
            ["Métrica", "Antes", "Después"],
            ["Número de filas", int(df.shape[0]), int(df_processed.shape[0])],
            ["Número de columnas", int(df.shape[1]), int(df_processed.shape[1])],
            ["Variable objetivo", "N/A", str(target_variable)],
            ["Características numéricas", "N/A", int(len(numeric_features))],
            ["Características categóricas", "N/A", int(len(categorical_features))],
            ["Valores nulos", int(df.isnull().sum().sum()), int(df_processed.isnull().sum().sum())]
        ]
        
        create_table_artifact(
            key="feature-engineering-summary",
            table=feature_summary,
            description="🔧 Resumen de ingeniería de características"
        )
        
        # Crear artefacto con lista de características
        features_table = [["Tipo", "Características"]]
        features_table.append(["Numéricas", ", ".join(str(f) for f in numeric_features)])
        features_table.append(["Categóricas", ", ".join(str(f) for f in categorical_features)])
        features_table.append(["Variable Objetivo", str(target_variable)])
        
        create_table_artifact(
            key="features-list",
            table=features_table,
            description="Lista de características por tipo"
        )
        
        # Crear artefacto con estadísticas básicas de las características numéricas
        if numeric_features:
            numeric_stats = df_processed[numeric_features].describe().round(4)
            stats_table = [["Estadística"] + [str(f) for f in numeric_features]]
            for stat in numeric_stats.index:
                row = [str(stat)] + [str(float(numeric_stats.loc[stat, col])) for col in numeric_features]
                stats_table.append(row)
            
            create_table_artifact(
                key="numeric-features-stats",
                table=stats_table,
                description="Estadísticas de características numéricas"
            )
        
        # Crear artefacto con distribución de la variable objetivo
        if target_variable in df_processed.columns:
            target_dist = df_processed[target_variable].value_counts()
            target_table = [["Valor", "Frecuencia", "Porcentaje"]]
            for value, count in target_dist.items():
                percentage = (count / len(df_processed)) * 100
                target_table.append([str(value), str(int(count)), f"{float(percentage):.2f}%"])
            
            create_table_artifact(
                key="target-distribution",
                table=target_table,
                description="Distribución de la variable objetivo"
            )
        
        return df_processed, target_variable, numeric_features, categorical_features
        
    except Exception as e:
        task_logger.error(f"Error en ingeniería de características: {str(e)}")
        raise


@task(name="train_model", retries=1, retry_delay_seconds=5)
def train_model(
    df: pd.DataFrame,
    target_variable: str,
    numeric_features: List[str],
    categorical_features: List[str],
    model_params: Optional[Dict[str, Any]] = None,
    optimization_params: Optional[Dict[str, Any]] = None,
    mlflow_config: Optional[Dict[str, Any]] = None
) -> Tuple[Any, str, Any, Dict[str, Any]]:

    task_logger = get_run_logger()
    task_logger.info("Iniciando entrenamiento del modelo")
    
    try:
        # Configuración por defecto de parámetros
        if model_params is None:
            model_params = {
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
        
        if optimization_params is None:
            optimization_params = {
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3, True),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float', 0.0, 1.0),
                'reg_lambda': ('float', 1.0, 10.0)
            }
        
        # Configurar MLflow
        if mlflow_config:
            if 'experiment_name' in mlflow_config:
                mlflow.set_experiment(mlflow_config['experiment_name'])
            if 'tracking_uri' in mlflow_config:
                mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        
        # Crear artefacto con configuración del modelo
        config_table = [["Parámetro", "Valor"]]
        config_table.append(["Algoritmo", "XGBoost Classifier"])
        config_table.append(["Test Size", "0.2"])
        config_table.append(["Métrica de optimización", "accuracy"])
        config_table.append(["Número de trials", str(int(mlflow_config.get('n_trials', 10) if mlflow_config else 10))])
        
        for param, value in model_params.items():
            config_table.append([f"model_{param}", str(value)])
        
        create_table_artifact(
            key="model-configuration",
            table=config_table,
            description="Configuración del modelo"
        )
        
        # Instanciar el entrenador
        trainer = TrainModel(
            df=df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_column=target_variable,
            model_class=XGBClassifier,
            test_size=0.2,
            model_params=model_params,
            param_distributions=optimization_params,
            n_trials=mlflow_config.get('n_trials', 10) if mlflow_config else 10,
            optimization_metric='accuracy',
            mlflow_setup=mlflow,
            mlflow_registered_model_name=mlflow_config.get('model_name') if mlflow_config else None
        )
        
        # Entrenar el modelo
        best_pipeline, best_run_id, study = trainer.train_with_optuna()
        
        task_logger.info(f"Entrenamiento completado exitosamente")
        task_logger.info(f"Mejor score: {study.best_value:.4f}")
        task_logger.info(f"MLflow run ID: {best_run_id}")
        
        # Calcular métricas detalladas en conjuntos de train y test
        task_logger.info("Calculando métricas detalladas para train y test...")
        
        # Obtener el pipeline entrenado y hacer predicciones
        X = df[numeric_features + categorical_features]
        y = df[target_variable]
        
        # Dividir en train y test (mismo split que en TrainModel)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Hacer predicciones en ambos conjuntos
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)
        y_test_pred_proba = best_pipeline.predict_proba(X_test)
        
        # Calcular métricas para conjunto de train
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        
        # Calcular métricas para conjunto de test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        task_logger.info(f"Métricas del modelo:")
        task_logger.info(f"Train - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        task_logger.info(f"Test - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        # Crear artefacto con métricas comparativas de train vs test
        metrics_comparison_table = [
            ["Métrica", "Train", "Test", "Diferencia", "Descripción"],
            ["Accuracy", f"{float(train_accuracy):.4f}", f"{float(test_accuracy):.4f}", f"{float(train_accuracy - test_accuracy):+.4f}", "Porcentaje de predicciones correctas"],
            ["F1-Score", f"{float(train_f1):.4f}", f"{float(test_f1):.4f}", f"{float(train_f1 - test_f1):+.4f}", "Media armónica de precisión y recall"],
            ["Precision", f"{float(train_precision):.4f}", f"{float(test_precision):.4f}", f"{float(train_precision - test_precision):+.4f}", "Proporción de positivos predichos correctos"],
            ["Recall", f"{float(train_recall):.4f}", f"{float(test_recall):.4f}", f"{float(train_recall - test_recall):+.4f}", "Proporción de positivos reales identificados"]
        ]
        
        create_table_artifact(
            key="model-metrics-comparison",
            table=metrics_comparison_table,
            description="Comparación de métricas: Train vs Test"
        )
        
        # Crear artefacto con métricas detalladas solo de test (para compatibilidad)
        test_metrics_table = [
            ["Métrica", "Valor", "Descripción"],
            ["Accuracy", f"{float(test_accuracy):.4f}", "Porcentaje de predicciones correctas"],
            ["F1-Score", f"{float(test_f1):.4f}", "Media armónica de precisión y recall"],
            ["Precision", f"{float(test_precision):.4f}", "Proporción de positivos predichos correctos"],
            ["Recall", f"{float(test_recall):.4f}", "Proporción de positivos reales identificados correctamente"]
        ]
        
        create_table_artifact(
            key="model-metrics-test",
            table=test_metrics_table,
            description="Métricas detalladas del conjunto de Test"
        )
        
        # Crear matriz de confusión (usando datos de test)
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        # Convertir matriz de confusión a tabla para el artefacto
        unique_labels = sorted(y_test.unique())
        conf_matrix_table = [["Predicho \\ Real"] + [str(label) for label in unique_labels]]
        
        for i, label in enumerate(unique_labels):
            row = [str(label)] + [str(int(conf_matrix[i][j])) for j in range(len(unique_labels))]
            conf_matrix_table.append(row)
        
        create_table_artifact(
            key="confusion-matrix",
            table=conf_matrix_table,
            description="Matriz de confusión del modelo (conjunto de Test)"
        )
        
        # Crear reporte de clasificación detallado (usando datos de test)
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Convertir reporte de clasificación a tabla
        class_report_table = [["Clase", "Precision", "Recall", "F1-Score", "Support"]]
        
        for label, metrics in class_report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                class_report_table.append([
                    str(label),
                    f"{float(metrics['precision']):.4f}",
                    f"{float(metrics['recall']):.4f}",
                    f"{float(metrics['f1-score']):.4f}",
                    str(int(metrics['support']))
                ])
        
        # Añadir métricas promedio
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in class_report:
                metrics = class_report[avg_type]
                class_report_table.append([
                    avg_type,
                    f"{float(metrics['precision']):.4f}",
                    f"{float(metrics['recall']):.4f}",
                    f"{float(metrics['f1-score']):.4f}",
                    str(int(metrics['support']))
                ])
        
        create_table_artifact(
            key="classification-report",
            table=class_report_table,
            description="Reporte de clasificación por clase (conjunto de Test)"
        )
        
        # Crear artefacto con resultados del entrenamiento (actualizado con todas las métricas)
        training_results = [
            ["Métrica", "Valor"],
            ["Mejor Score (Accuracy Optuna)", f"{float(study.best_value):.4f}"],
            ["Accuracy (Train Set)", f"{float(train_accuracy):.4f}"],
            ["Accuracy (Test Set)", f"{float(test_accuracy):.4f}"],
            ["F1-Score (Train Set)", f"{float(train_f1):.4f}"],
            ["F1-Score (Test Set)", f"{float(test_f1):.4f}"],
            ["Precision (Train Set)", f"{float(train_precision):.4f}"],
            ["Precision (Test Set)", f"{float(test_precision):.4f}"],
            ["Recall (Train Set)", f"{float(train_recall):.4f}"],
            ["Recall (Test Set)", f"{float(test_recall):.4f}"],
            ["Número de trials ejecutados", str(int(len(study.trials)))],
            ["MLflow Run ID", str(best_run_id)],
            ["Modelo registrado", str(mlflow_config.get('model_name', 'N/A') if mlflow_config else 'N/A')]
        ]
        
        create_table_artifact(
            key="training-results",
            table=training_results,
            description="Resultados del entrenamiento"
        )
        
        # Crear artefacto con mejores hiperparámetros
        best_params_table = [["Hiperparámetro", "Mejor Valor"]]
        for param, value in study.best_params.items():
            best_params_table.append([param, str(value)])
        
        create_table_artifact(
            key="best-hyperparameters",
            table=best_params_table,
            description="Mejores hiperparámetros encontrados"
        )
        
        # Preparar diccionario de métricas para retornar
        metrics_dict = {
            'accuracy': float(test_accuracy),
            'f1_score': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        return best_pipeline, best_run_id, study, metrics_dict
        
    except Exception as e:
        task_logger.error(f"Error en el entrenamiento: {str(e)}")
        raise


@flow(
    name="ml_pipeline",
    description="Pipeline completo: ETL -> Feature Engineering -> Training"
)
def ml_pipeline_flow(
    data_file_path: str = "data/raw_data/BD_ordenes.xlsx",
    model_settings: Optional[Dict[str, Any]] = None,
    mlflow_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    logger.info("=== INICIANDO PIPELINE ===")
    
    # Configuración por defecto
    if model_settings is None:
        model_settings = {
            'model_params': {
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'verbosity': 0
            },
            'optimization_params': {
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3, True),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float', 0.0, 1.0),
                'reg_lambda': ('float', 1.0, 10.0)
            }
        }
    
    if mlflow_config is None:
        mlflow_config = {
            'experiment_name': 'ml_pipeline_experiment',
            'tracking_uri': './mlruns',
            'n_trials': 10,
            'model_name': 'energy_consumption_model'
        }
    
    try:
        # Paso 1: Extracción y carga de datos
        logger.info("PASO 1: Extracción y carga de datos")
        raw_data = extract_and_load_data(data_file_path)
        
        # Paso 2: Ingeniería de características
        logger.info("PASO 2: Ingeniería de características")
        processed_data, target_var, numeric_feat, categorical_feat = feature_engineering(raw_data)
        
        # Paso 3: Entrenamiento del modelo
        logger.info("PASO 3: Entrenamiento del modelo")
        pipeline, run_id, study, metrics = train_model(
            df=processed_data,
            target_variable=target_var,
            numeric_features=numeric_feat,
            categorical_features=categorical_feat,
            model_params=model_settings['model_params'],
            optimization_params=model_settings['optimization_params'],
            mlflow_config=mlflow_config
        )
        
        # Compilar resultados
        results = {
            'pipeline': pipeline,
            'mlflow_run_id': run_id,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'data_shape': processed_data.shape,
            'target_variable': target_var,
            'numeric_features': numeric_feat,
            'categorical_features': categorical_feat,
            'n_trials': len(study.trials),
            'metrics': metrics
        }
        
        # Crear artefacto final con resumen completo del pipeline
        pipeline_summary = [
            ["Fase", "Resultado"],
            ["Datos originales", f"{int(raw_data.shape[0])} filas, {int(raw_data.shape[1])} columnas"],
            ["Datos procesados", f"{int(processed_data.shape[0])} filas, {int(processed_data.shape[1])} columnas"],
            ["Variable objetivo", str(target_var)],
            ["Características numéricas", f"{int(len(numeric_feat))} variables"],
            ["Características categóricas", f"{int(len(categorical_feat))} variables"],
            ["Modelo utilizado", "XGBoost Classifier"],
            ["Mejor score (Optuna)", f"{float(study.best_value):.4f}"],
            ["Accuracy (Test)", f"{float(metrics['accuracy']):.4f}"],
            ["F1-Score (Test)", f"{float(metrics['f1_score']):.4f}"],
            ["Precision (Test)", f"{float(metrics['precision']):.4f}"],
            ["Recall (Test)", f"{float(metrics['recall']):.4f}"],
            ["Trials ejecutados", f"{int(len(study.trials))}"],
            ["MLflow Run ID", str(run_id)]
        ]
        
        create_table_artifact(
            key="pipeline-summary",
            table=pipeline_summary,
            description="Resumen completo del pipeline de ML"
        )
        
        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
        logger.info(f"Mejor score obtenido: {results['best_score']:.4f}")
        logger.info(f"MLflow Run ID: {results['mlflow_run_id']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}")
        raise

# Función para ejecutar el pipeline
def run_pipeline(
    data_file_path: str = "data/raw_data/BD_ordenes.xlsx",
    experiment_name: str = "ml_pipeline_experiment",
    model_name: str = "energy_consumption_model",
    n_trials: int = 10,
    tracking_uri: str = "./mlruns"
) -> Dict[str, Any]:
    
    # Configuración de MLflow
    mlflow_config = {
        'experiment_name': experiment_name,
        'tracking_uri': tracking_uri,
        'n_trials': n_trials,
        'model_name': model_name
    }
    
    # Configuración del modelo
    model_settings = {
        'model_params': {
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'verbosity': 0
        },
        'optimization_params': {
            'n_estimators': ('int', 100, 400),
            'max_depth': ('int', 3, 12),
            'learning_rate': ('float', 0.01, 0.3, True),
            'subsample': ('float', 0.7, 1.0),
            'colsample_bytree': ('float', 0.7, 1.0),
            'reg_alpha': ('float', 0.0, 1.0),
            'reg_lambda': ('float', 1.0, 10.0),
            'min_child_weight': ('int', 1, 6)
        }
    }
    
    # Ejecutar el flujo
    return ml_pipeline_flow(
        data_file_path=data_file_path,
        model_settings=model_settings,
        mlflow_config=mlflow_config
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict with Prefect + MLflow.')
    parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI (overrides environment variable)')
    args = parser.parse_args()

    # Override MLflow URI if provided
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
        setup_mlflow()

    try:
        # Run the flow
        run_id = run_pipeline()
        logger.info("\n Pipeline completed successfully!")
        logger.info(f"MLflow run_id: {run_id}")
        logger.info(f"View results at: {mlflow.get_tracking_uri()}")

        # Save run ID for reference
        #with open("prefect_run_id.txt", "w") as f:
        #    f.write(run_id)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
