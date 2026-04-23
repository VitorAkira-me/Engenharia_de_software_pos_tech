# exemplos/refatoracao/depois/config.py
"""Configurações centralizadas do projeto."""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configurações da aplicação."""
    
    # Caminhos
    data_path: Path = Path('./data/customer_data.csv')
    output_dir: Path = Path('./output')
    
    # Parâmetros do modelo
    test_size: float = 0.3
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    
    # Parâmetros de features
    age_bins: list = None
    age_labels: list = None
    high_income_threshold: int = 100000
    
    def __post_init__(self):
        """Inicializa valores padrão complexos."""
        if self.age_bins is None:
            self.age_bins = [0, 25, 45, 65, 120]
        if self.age_labels is None:
            self.age_labels = ['young', 'adult', 'middle', 'senior']
        
        # Garante que diretórios existem
        self.output_dir.mkdir(parents=True, exist_ok=True)