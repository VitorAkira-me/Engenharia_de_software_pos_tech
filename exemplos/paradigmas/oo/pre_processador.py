from typing import Optional, Tuple

import pandas as pd


class PreProcessador:
    """Responsável por limpeza e preparação de dados."""
    def __init__(self, target_col: str = 'preco'):
        self.target_col = target_col
        self.feature_cols: Optional[list] = None
        
    def processar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar limpeza completa"""
        df = self._remover_nulos(df)
        df = self._remover_outliers(df)
        self._definir_features(df)
        return df
    
    def _remover_nulos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros com valores nulos."""
        n_antes = len(df)
        df = df.dropna()
        n_depois = len(df)
        print(f"Nulos removidos: {n_antes - n_depois} registros")
        return df
    
    def _remover_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove vaslores negativos."""
        df = df[df['area'] > 0]
        df = df[df[self.target_col] > 0]
        return df
    
    def _definir_features(self, df: pd.DataFrame) -> None:
        """Define colunas de features."""
        self.feature_cols = [col for col in df.columns if col != self.target_col]

    def separar_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separa X e y."""
        if self.feature_cols is None:
            self._definir_features(df)

        X = df[self.feature_cols]
        y = df[self.target_col]
        return X, y