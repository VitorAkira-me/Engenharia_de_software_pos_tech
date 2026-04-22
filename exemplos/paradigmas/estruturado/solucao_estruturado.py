"""Solução usando programação estruturada (funções)."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple

# Variáveis globais para manter estado (problemático!)
model = None
feature_cols = None
is_trained = False

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega dados de CSV."""
    df = pd.read_csv(caminho)
    print(f"Dados carregados: {len(df)} registros")
    return df


def pre_processar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e prepara dados."""
    # Remove nulos
    df = df.dropna()

    # Remove outliers (valores negativos)
    df = df[df['area'] > 0]
    df = df[df['preco'] > 0]

    print(f"Após pré-processamento: {len(df)} registros")
    return df

def separar_features_target(df: pd.DataFrame, target_col: str = 'preco') -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features e target"""
    global feature_cols

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    return X, y

def dividir_treino_teste(X: pd.DataFrame, y: pd.Series,
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple:
    """Divide dados em treino e teste."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def treinar_modelo(X_train: pd.DataFrame, y_train: pd.Series,
                   n_estimators: int = 100,
                   random_state: int = 42) -> None:
    """Treina modelo RandomForest"""
    global model, is_trained

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    print("Treinando modelo...")
    model.fit(X_train, y_train)
    is_trained = True
    print("Modelo treinado!")

def fazer_predicao(X: pd.DataFrame) -> np.ndarray:
    """Faz predição usando modelo treinado."""
    global model, is_trained, feature_cols

    if not is_trained or model is None:
        raise ValueError("Modelo não foi treinado ainda!")
    
    # Garante ordem correta das features
    X = X[feature_cols]
    return model.predict(X)

def avaliar_modelo(X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Avalia performance do modelo."""
    if not is_trained:
        raise ValueError("Modelo não foi treinado ainda!")
    
    y_pred = fazer_predicao(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metricas = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    return metricas

def pipeline_completo(caminho_dados: str) -> dict:
    """Executa pipeline completo."""
    # Carrega
    df = carregar_dados(caminho_dados)

    # Pré-processa
    df = pre_processar(df)

    # Separa features e target
    X, y = separar_features_target(df)

    #Divide treino/teste
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)

    # Treina
    treinar_modelo(X_train, y_train)

    # Avalia
    metricas = avaliar_modelo(X_test, y_test)

    print("\n=== Métricas ===")
    print(f"RMSE: ${metricas['rmse']:,.2f}")
    print(f"R²: {metricas['r2']:.3f}")

    return metricas

if __name__ == "__main__":
    metricas = pipeline_completo('imoveis.csv')

    # Exemplo de predição nova
    novo_imovel = pd.DataFrame({
        'area': [200],
        'quartos': [3],
        'banheiros': [2],
        'idade': [5],
        'distancia_centro': [8]
    })

    preco_previsto = fazer_predicao(novo_imovel)
    print(f"\nPreço previsto para novo imóvel: ${preco_previsto[0]:,.2f}")