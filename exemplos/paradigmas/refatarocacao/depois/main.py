# exemplos/refatoracao/depois/main.py
"""Ponto de entrada da aplicação."""
import logging
from config import Config
from pipeline import ChurnAnalysisPipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Função principal."""
    # Carrega configurações
    config = Config()
    
    # Cria e executa pipeline
    pipeline = ChurnAnalysisPipeline(config)
    results = pipeline.run()
    
    # Exemplo de predição
    print("\n=== Predição para Novo Cliente ===")
    novo_cliente = {
        'age': 35,
        'income': 75000,
        'education': 'bachelor',
        'customer_id': 99999
    }
    
    prediction = pipeline.predict_customer(novo_cliente)
    print(f"Cliente com {novo_cliente['age']} anos, income ${novo_cliente['income']:,}")
    print(f"Predição: {prediction}")

if __name__ == "__main__":
    main()