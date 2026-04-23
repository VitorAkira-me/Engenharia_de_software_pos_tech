# exemplos/refatoracao/depois/data_cleaner.py
"""Módulo responsável por limpeza de dados."""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Aplica regras de limpeza nos dados."""
    
    def __init__(self, age_min: int = 0, age_max: int = 120):
        """
        Inicializa cleaner.
        
        Args:
            age_min: Idade mínima válida
            age_max: Idade máxima válida
        """
        self.age_min = age_min
        self.age_max = age_max
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as regras de limpeza.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame limpo
        """
        n_original = len(df)
        
        df = self._remove_duplicates(df)
        df = self._remove_nulls(df)
        df = self._fix_age(df)
        df = self._fix_income(df)
        df = self._fix_education(df)
        
        n_final = len(df)
        logger.info(f"Limpeza concluída: {n_original} → {n_final} registros ({n_original - n_final} removidos)")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros duplicados."""
        return df.drop_duplicates()
    
    def _remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linhas com nulos em colunas críticas."""
        critical_cols = ['customer_id', 'age', 'churn']
        return df.dropna(subset=critical_cols)
    
    def _fix_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige valores de idade."""
        if 'age' not in df.columns:
            return df
        
        # Filtra idades válidas
        df = df[(df['age'] > self.age_min) & (df['age'] < self.age_max)]
        return df
    
    def _fix_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige valores de income."""
        if 'income' not in df.columns:
            return df
        
        # Remove valores negativos
        df = df[df['income'] > 0]
        
        # Preenche nulos com mediana
        if df['income'].isnull().any():
            median_income = df['income'].median()
            df['income'] = df['income'].fillna(median_income)
            logger.info(f"Income nulos preenchidos com mediana: ${median_income:,.2f}")
        
        return df
    
    def _fix_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige valores de education."""
        if 'education' not in df.columns:
            return df
        
        df['education'] = df['education'].fillna('unknown')
        return df