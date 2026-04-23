# exemplos/refatoracao/depois/churn_predictor.py
"""Módulo responsável pelo modelo de predição de churn."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Modelo de predição de churn de clientes."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 5, random_state: int = 42):
        """
        Inicializa preditor.
        
        Args:
            n_estimators: Número de árvores
            max_depth: Profundidade máxima das árvores
            min_samples_split: Mínimo de samples para split
            random_state: Seed para reprodutibilidade
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self._is_trained = False
        self._feature_names: Optional[list] = None
    
    @property
    def is_trained(self) -> bool:
        """Verifica se modelo está treinado."""
        return self._is_trained
    
    @property
    def feature_names(self) -> list:
        """Retorna nomes das features."""
        if not self._is_trained:
            raise ValueError("Modelo não treinado")
        return self._feature_names
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Treina o modelo.
        
        Args:
            X: Features de treino
            y: Target de treino
        """
        logger.info(f"Treinando modelo com {len(X)} samples...")
        
        self._feature_names = list(X.columns)
        self.model.fit(X, y)
        self._is_trained = True
        
        logger.info("Treinamento concluído")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições.
        
        Args:
            X: Features para predição
            
        Returns:
            Array de predições
        """
        if not self._is_trained:
            raise ValueError("Modelo não está treinado")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades das predições."""
        if not self._is_trained:
            raise ValueError("Modelo não está treinado")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Avalia performance do modelo.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Acurácia: {metrics['accuracy']:.3f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna importância das features.
        
        Returns:
            DataFrame com features e importâncias
        """
        if not self._is_trained:
            raise ValueError("Modelo não está treinado")
        
        importances = self.model.feature_importances_
        
        df_importance = pd.DataFrame({
            'feature': self._feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_importance