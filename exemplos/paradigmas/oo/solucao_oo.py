# exemplos/paradigmas/oo/solucao_oo.py
"""Solução usando Programação Orientada a Objetos."""
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processador import PreProcessador
from modelo_preditivo import ModeloPreditivo
from pipeline_ml import PipelineML

if __name__ == "__main__":
    # Exemplo 1: pipeline simples
    print("=== EXEMPLO 1: Pipeline Único ===")

    preprocessador = PreProcessador(target_col='preco')
    modelo = ModeloPreditivo(nome="RandomForest_v1", n_estimators=100)

    pipeline = PipelineML(preprocessador, modelo)
    metricas = pipeline.executar('imoveis.csv')

    # Predição em novo dado
    novo_imovel = pd.DataFrame({
        'area': [200],
        'quartos': [3],
        'banheiros': [2],
        'idade': [5],
        'distancia_centro': [8]
    })

    preco = modelo.prever(novo_imovel)
    print(f"\nPreço previsto: ${preco[0]:,.2f}")
    print(f"\nInfo do modelo: {modelo.info()}")

    # Exemplo 2: MÚLTIPLOS MODELOS simultaneamente (impossível na versão funcional!)
    print("\n\n=== EXEMPLO 2: Múltiplos Modelos ===")

    # Carrega e prepara dados uma vez
    df = pd.read_csv('imoveis.csv')
    prep = PreProcessador()
    df = prep.processar(df)
    X, y = prep.separar_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria 3 modelos diferentes
    modelo1 = ModeloPreditivo(nome="RF_50", n_estimators=50)
    modelo2 = ModeloPreditivo(nome="RF_100", n_estimators=100)
    modelo3 = ModeloPreditivo(nome="RF_200", n_estimators=200)

    #Treina todos
    for modelo in [modelo1, modelo2, modelo3]:
        modelo.treinar(X_train, y_train)
        metricas = modelo.avaliar(X_test, y_test)
        print(f"{modelo.nome} - RMSE: ${metricas['rmse']:,.2f},R²: {metricas['r2']:.3f}")

    # Compara predições
    print("\nPredições do mesmo imóvel por diferentes modelos:")
    for modelo in [modelo1, modelo2, modelo3]:
        pred = modelo.prever(novo_imovel)
        print(f"{modelo.nome}: ${pred[0]:,.2f}")