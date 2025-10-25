import pandas as pd
import numpy as np

class FeatureEngineering:
    """Feature Engineer.

        Esta clase se encarga de la preparación de características específicas para el análisis de datos.

    Attributes:
        df (pd.DataFrame): El DataFrame que contiene los datos.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _generic_feature_preparation(self) -> pd.DataFrame:
        """ Prepara características genéricas para el análisis de datos.

        Args:
            df (pd.DataFrame): El DataFrame que contiene los datos.

        Returns:
            pd.DataFrame: El DataFrame con las características genéricas preparadas.
        """
        # Crear nueva columna 'tipo_servicio'
        mapeo_tipo_servicio = {
            '701-ENERGÍA MDO REGULADO': 'Energia',
            '101-AGUA POTABLE': 'Agua',
            '103-ALCANTARILLADO': 'Alcantarillado',
            '501-GAS NATURAL REGULADO': 'Gas',
            '1007-ALUMBRADO PÚBLICO MEDIDO': 'Energia',
            '8000-AGUA POTABLE OCC.': 'Agua',
            '8003-AGUA POTABLE URABA': 'Agua',
            '7505-GAS NATURAL COMPRIMIDO (GNC)': 'Gas',
            '240-AGUA POTABLE MALAMBO': 'Agua',
            '249-AGUA POTABLE DE RIONEGRO ANT': 'Agua',
            '1702-MOVILIDAD ELÉCTRICA CARGA INTE': 'Energia'
        }

        df_generic_fp = self.df.copy()

        df_generic_fp['tipo_servicio'] = df_generic_fp['servicio'].map(mapeo_tipo_servicio)
        df_generic_fp.drop(columns=['servicio'], inplace=True)

        # Intercambiar el valor de la variable respuesta
        df_generic_fp['respuesta'] = 1 - df_generic_fp['respuesta']

        return df_generic_fp
    
    def _list_target_numeric_and_categorical_features (self, df: pd.DataFrame):
        """ Obtiene la lista de variables categoricas y numericas de un dataframe

        Args:
             df (pd.DataFrame): El DataFrame que contiene los datos.

        Returns:
            target_variable: variable a predecir
            numeric_features: Lista con las variables numericas
            categorical_features: Lista con las variables categoricas
        """
        target_variable = 'respuesta'
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_features = [col for col in numeric_cols if col != target_variable]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_features = [col for col in categorical_cols]
        
        return target_variable, numeric_features, categorical_features


    def prepare_features_only_energy(self) -> pd.DataFrame:
        """ Prepara las características específicas para el análisis de datos de energía.

        Returns:
            pd.DataFrame: El DataFrame con las características preparadas de datos de energía.
        """

        df_energy = self._generic_feature_preparation()
        df_energy = df_energy[df_energy['tipo_servicio'] == 'Energia']
        target_variable, numeric_features, categorical_features = self._list_target_numeric_and_categorical_features(df_energy)


        return df_energy,target_variable, numeric_features, categorical_features