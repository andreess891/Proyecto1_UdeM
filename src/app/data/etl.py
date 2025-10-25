import pandas as pd

class GetData:
    """Obtener los datos.

    Esta clase se encarga de la extracción y transformación de los datos desde un archivo excel.

    Attributes:
        file_path (str): La ruta del archivo excel que contiene los datos.
    """

    def __init__(self, file_path: str = "raw_data/BD_ordenes.xlsx"):
        self.file_path = file_path
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mapea las columnas del DataFrame a nombres más amigables.

        Args:
            df (pd.DataFrame): El DataFrame original.

        Returns:
            pd.DataFrame: El DataFrame con las columnas renombradas.
        """
        column_mapping = {
            'RESPUESTA': 'respuesta', 
            'CONSUMO_CRITICADO': 'consumo_criticado', 
            'SERVICIO': 'servicio',
            'CATEGORIA': 'categoria', 
            'NIVEL_TENSION': 'nivel_tension', 
            'ESTRATO': 'estrato', 
            'LOCALIDAD': 'localidad', 
            'FUNCION_ANALISIS': 'funcion_analisis',
            'CALIFICACION': 'calificacion',
            'OBS_LECTURA': 'obs_lectura',
            'PERIODICIDAD': 'periodicidad'
        }

        return df.rename(columns=column_mapping)

    def read_data(self) -> pd.DataFrame:
        """Lee los datos desde el archivo excel y mapea las columnas.

        Args:
            file_path (str): La ruta del archivo excel que contiene los datos.

        Returns:
            pd.DataFrame: El DataFrame con los datos leídos y las columnas mapeadas.
        """
        df = pd.read_excel(self.file_path)
        return self._map_columns(df)
