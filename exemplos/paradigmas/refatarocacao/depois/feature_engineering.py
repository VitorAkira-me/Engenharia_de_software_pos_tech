# exemplos/refatoracao/depois/feature_engineer.py
"""Módulo responsável por feature engineering."""
import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Cria features derivadas dos dados."""
    
    def __init__(self, age_bins: List[int], age_labels: List[str], 
                 high_income_threshold: int):
        """
        Inicializa engineer.
        
        Args:
            age_bins: Limites para binning de idade
            age_labels: Labels para grupos de idade
            high_income_threshold: Threshold para flag de alto income
        """
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.high_income_threshold = high_income_threshold
        self.feature_names: List[str] = []
    
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as transformações de features.
        
        Args:
            df: DataFrame com dados limpos
            
        Returns:
            DataFrame com features engineered
        """
        logger.info("Iniciando feature engineering...")
        
        df = self._create_ratio_features(df)
        df = self._create_age_groups(df)
        df = self._create_binary_flags(df)
        df = self._encode_categorical(df)
        
        self._define_feature_names(df)
        
        logger.info(f"Features criadas: {len(self.feature_names)} features")
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de razão."""
        if 'income' in df.columns and 'age' in df.columns:
            df['income_age_ratio'] = df['income'] / df['age']
        return df
    
    def _create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria grupos etários."""
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=self.age_bins, labels=self.age_labels)
        return df
    
    def _create_binary_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria flags binárias."""
        if 'income' in df.columns:
            df['high_value'] = (df['income'] > self.high_income_threshold).astype(int)
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica one-hot encoding."""
        categorical_cols = ['education', 'age_group']
        existing_cols = [col for col in categorical_cols if col in df.columns]
        
        if existing_cols:
            df = pd.get_dummies(df, columns=existing_cols, drop_first=False)
        
        return df
    
    def _define_feature_names(self, df: pd.DataFrame) -> None:
        """Define lista de nomes de features."""
        exclude = ['customer_id', 'churn']
        self.feature_names = [col for col in df.columns if col not in exclude]
    
    def transform_new_data(self, df_new: pd.DataFrame, 
                          reference_features: List[str]) -> pd.DataFrame:
        """
        Transforma novos dados garantindo compatibilidade.
        
        Args:
            df_new: Novos dados
            reference_features: Features esperadas (do treino)
            
        Returns:
            DataFrame transformado
        """
        # Aplica as mesmas transformações
        df_new = self._create_ratio_features(df_new)
        df_new = self._create_age_groups(df_new)
        df_new = self._create_binary_flags(df_new)
        df_new = self._encode_categorical(df_new)
        
        # Garante mesmas colunas que o treino
        for col in reference_features:
            if col not in df_new.columns:
                df_new[col] = 0
        
        # Remove colunas extras e reordena
        df_new = df_new[reference_features]
        
        return df_new