# exemplos/refatoracao/depois/pipeline.py
"""Pipeline principal que orquestra todo o processo."""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Dict

from config import Config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from churn_predictor import ChurnPredictor
from report_generator import ReportGenerator

logger = logging.getLogger(__name__)

class ChurnAnalysisPipeline:
    """Orquestra todo o processo de análise de churn."""
    
    def __init__(self, config: Config):
        """
        Inicializa pipeline.
        
        Args:
            config: Objeto de configuração
        """
        self.config = config
        
        # Inicializa componentes
        self.loader = DataLoader(config.data_path)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer(
            age_bins=config.age_bins,
            age_labels=config.age_labels,
            high_income_threshold=config.high_income_threshold
        )
        self.predictor = ChurnPredictor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            random_state=config.random_state
        )
        self.reporter = ReportGenerator(config.output_dir)
    
    def run(self) -> Dict:
        """
        Executa pipeline completo.
        
        Returns:
            Dicionário com resultados e métricas
        """
        logger.info("=== Iniciando Pipeline de Análise de Churn ===")
        
        # 1. Carrega dados
        df = self.loader.load()
        
        # 2. Limpa dados
        df = self.cleaner.clean(df)
        
        # 3. Feature engineering
        df = self.engineer.engineer(df)
        
        # 4. Separa features e target
        X = df[self.engineer.feature_names]
        y = df['churn']
        
        # 5. Divide treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        logger.info(f"Dados divididos: {len(X_train)} treino, {len(X_test)} teste")
        
        # 6. Treina modelo
        self.predictor.train(X_train, y_train)
        
        # 7. Avalia
        metrics = self.predictor.evaluate(X_test, y_test)
        
        # 8. Gera relatórios
        summary = self.reporter.generate_summary(df)
        self.reporter.print_summary(summary)
        
        importance = self.predictor.get_feature_importance()
        self.reporter.plot_feature_importance(importance)
        
        logger.info("=== Pipeline concluído com sucesso ===")
        
        return {
            'metrics': metrics,
            'summary': summary,
            'feature_importance': importance
        }
    
    def predict_customer(self, customer_data: Dict) -> str:
        """
        Prediz churn para um novo cliente.
        
        Args:
            customer_data: Dicionário com dados do cliente
            
        Returns:
            Predição ('Churn' ou 'No Churn')
        """
        if not self.predictor.is_trained:
            raise ValueError("Modelo não foi treinado. Execute run() primeiro.")
        
        # Cria DataFrame
        df_new = pd.DataFrame([customer_data])
        
        # Aplica mesmas transformações
        df_new = self.engineer.transform_new_data(
            df_new, 
            self.predictor.feature_names
        )
        
        # Prediz
        prediction = self.predictor.predict(df_new)[0]
        proba = self.predictor.predict_proba(df_new)[0]
        
        result = 'Churn' if prediction == 1 else 'No Churn'
        confidence = proba[1] if prediction == 1 else proba[0]
        
        logger.info(f"Predição: {result} (confiança: {confidence:.2%})")
        
        return result