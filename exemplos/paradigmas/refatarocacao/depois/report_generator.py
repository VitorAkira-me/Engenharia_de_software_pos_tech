# exemplos/refatoracao/depois/report_generator.py
"""Módulo responsável por geração de relatórios."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Gera relatórios e visualizações."""
    
    def __init__(self, output_dir: Path):
        """
        Inicializa gerador.
        
        Args:
            output_dir: Diretório para salvar outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """
        Gera resumo estatístico dos dados.
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Dicionário com estatísticas
        """
        summary = {
            'total_customers': len(df),
            'churn_rate': df['churn'].mean() if 'churn' in df.columns else None,
            'avg_income': df['income'].mean() if 'income' in df.columns else None,
            'avg_age': df['age'].mean() if 'age' in df.columns else None
        }
        
        logger.info("Resumo estatístico gerado")
        return summary
    
    def print_summary(self, summary: Dict) -> None:
        """Imprime resumo formatado."""
        print("\n=== RELATÓRIO DE ANÁLISE ===")
        print(f"Total de clientes: {summary['total_customers']:,}")
        
        if summary['churn_rate'] is not None:
            print(f"Taxa de churn: {summary['churn_rate']:.2%}")
        
        if summary['avg_income'] is not None:
            print(f"Income médio: ${summary['avg_income']:,.2f}")
        
        if summary['avg_age'] is not None:
            print(f"Idade média: {summary['avg_age']:.1f} anos")
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                                top_n: int = 15) -> Path:
        """
        Plota importância das features.
        
        Args:
            importance_df: DataFrame com features e importâncias
            top_n: Número de top features a mostrar
            
        Returns:
            Caminho do arquivo salvo
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importância')
        plt.title(f'Top {top_n} Features Mais Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = self.output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico salvo em: {output_path}")
        return output_path