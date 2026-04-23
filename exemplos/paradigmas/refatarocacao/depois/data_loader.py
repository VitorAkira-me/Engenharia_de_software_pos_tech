# exemplos/refatoracao/depois/data_loader.py
"""Módulo responsável por carregar dados."""
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Carrega dados de diferentes fontes."""
    
    def __init__(self, data_path: Path):
        """
        Inicializa loader.
        
        Args:
            data_path: Caminho para arquivo de dados
        """
        self.data_path = Path(data_path)
    
    def load(self) -> pd.DataFrame:
        """
        Carrega dados de CSV.
        
        Returns:
            DataFrame com dados carregados
            
        Raises:
            FileNotFoundError: Se arquivo não existe
            ValueError: Se arquivo está vazio ou mal formatado
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Dados carregados: {len(df)} registros de {self.data_path}")
            
            if df.empty:
                raise ValueError("Arquivo está vazio")
            
            return df
        
        except pd.errors.EmptyDataError:
            raise ValueError(f"Arquivo CSV vazio: {self.data_path}")
        except Exception as e:
            raise ValueError(f"Erro ao ler CSV: {e}")