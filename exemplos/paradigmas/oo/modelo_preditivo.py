from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class ModeloPreditivo:
    """Encapsula modelo de ML com seu estado e comportamento."""

    def __init__(self, nome: str = "ModeloPadrao",
                 n_estimators: int = 100,
                 random_state: int = 42):
        """
        Inicializa modelo.

        Args:
            nome: Identificador do modelo
            n_estimatos: Número de árvores do RandomForest
            random_state: Seed para reprodutibilidade
        """
        self.nome = nome
        self.n_estimators = n_estimators
        self.random_state = random_state

        # Estado interno
        self._modelo = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self._treinado = False
        self._feature_cols: Optional[list] = None
        self._data_treino: Optional[datetime] = None
        self._metricas_treino: Optional[dict] = None

    @property
    def esta_treinado(self) -> bool:
        """Verifica se modelo está treinado."""
        return self._treinado
    
    def treinar(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Treina o modelo."""
        print(f"[{self.nome}] Iniciando treinamento...")

        self._feature_cols = list(X_train.columns)
        self._modelo.fit(X_train, y_train)
        self._treinado = True
        self._data_treino = datetime.now()

        print(f"[{self.nome}] Treinamento concluído!")

    def prever(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições"""
        if not self._treinado:
            raise ValueError(f"[{self.nome}] Modelo não está treinado!")
        
        # Garante ordem das features
        X = X[self._feature_cols]
        return self._modelo.predict(X)
    
    def avaliar(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Avalia performance."""
        y_pred = self.prever(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metricas = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

        self._metricas_treino = metricas
        return metricas
    
    def info(self) -> dict:
        """Retorna informações do modelo."""
        return {
            'nome': self.nome,
            'treinado': self._treinado,
            'data_treino': self._data_treino,
            'n_features': len(self._feature_cols) if self._feature_cols else 0,
            'metricas': self._metricas_treino
        }
    
    def __repr__(self) -> str:
        status = "treinado" if self._treinado else "Não treinado"
        return f"ModeloPreditivo(nome='{self.nome}', status='{status}')"
    