# exemplos/paradigmas/gerar_dados.py
"""Gera dados sintéticos de preços de imóveis."""
import pandas as pd
import numpy as np
#pseudoaleatorio por isso colocamos o seed 42, para retornar a mesma coisa.
def gerar_dados_imoveis(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Gera dataset sintético de preços de imóveis."""
    np.random.seed(seed)

    # Features
    area = np.random.normal(150, 50, n_samples)
    quartos = np.random.randint(1, 6, n_samples)
    banheiros = np.random.randint(1, 4, n_samples)
    idade = np.random.randint(0, 50, n_samples)
    distancia_centro = np.random.normal(10, 5, n_samples)

    # Target (com alguma lógica)
    preco = (
        3000 * area +
        50000 * quartos +
        30000 * banheiros -
        1000 * idade -
        2000 * distancia_centro +
        np.random.normal(0, 50000, n_samples) 
    )

    df = pd.DataFrame({
        'area': area,
        'quartos': quartos,
        'banheiros': banheiros,
        'idade': idade,
        'distancia_centro': distancia_centro,
        'preco': preco
    })

    return df

if __name__== "__main__":
    df = gerar_dados_imoveis()
    df.to_csv('imoveis.csv', index=False)
    print(f"Dataset gerado: {len(df)} imóveis")
    print(df.head())