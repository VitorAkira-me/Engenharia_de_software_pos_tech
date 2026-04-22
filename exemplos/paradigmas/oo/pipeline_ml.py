import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processador import PreProcessador
from modelo_preditivo import ModeloPreditivo

class PipelineML:
    """Orquestra todo o processo de ML."""

    def __init__(self, preprocessador: PreProcessador, modelo: ModeloPreditivo):
        """
        Inicializa pipeline.

        Args:
            preprocessador: Instância de PreProcessador
            modelo: Instância de ModeloPreditivo
        """
        self.preprocessador = preprocessador
        self.modelo = modelo
        self.test_size = 0.2

    def executar(self, caminho_dados: str) -> dict:
        """Executa pipeline completo."""
        # 1. Carrega dados
        print("\n=== Carrega dados ===")
        df = pd.read_csv(caminho_dados)
        print(f"Registros carregados: {len(df)}")

        # 2. Pré-processa
        print("\n=== Pré-processando ===")
        df = self.preprocessador.processar(df)
        print(f"Registros após limpeza: {len(df)}")

        # 3. Separa features e target
        X, y = self.preprocessador.separar_features_target(df)

        # 4. Divide treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # 5. Treina modelo
        print("\n=== Treinando ===")
        self.modelo.treinar(X_train, y_train)

        # 6. Avalia
        print("\n=== Avaliando ===")
        metricas = self.modelo.avaliar(X_test, y_test)

        print(f"\nRMSE: ${metricas['rmse']:,.2f}")
        print(f"R²: {metricas['r2']:.3f}")

        return metricas