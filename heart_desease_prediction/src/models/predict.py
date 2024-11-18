import logging
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Dict, List
import yaml

from src.utils.helpers import setup_logging, load_config
from src.data.preprocess import preprocess_data

logger = logging.getLogger(__name__)
setup_logging()


class ModelPredictor:
    """Classe para fazer predições com modelos treinados"""

    def __init__(self, model_path: Union[str, Path], config_path: Union[str, Path]):
        """
        Inicializa o preditor

        Args:
            model_path: Caminho para o modelo salvo
            config_path: Caminho para arquivo de configuração
        """
        self.model_path = Path(model_path)
        self.config = load_config(config_path)
        self.model = self.load_model()

    def load_model(self):
        """Carrega o modelo salvo"""
        logger.info(f"Carregando modelo de {self.model_path}")
        return joblib.load(self.model_path)

    def predict_single(self, data: Dict[str, Union[float, str]]) -> float:
        """
        Faz predição para uma única amostra

        Args:
            data: Dicionário com os dados de entrada

        Returns:
            Probabilidade de doença cardíaca
        """
        # Converter dicionário para DataFrame
        df = pd.DataFrame([data])

        # Preprocessar dados
        processed_data = preprocess_data(df, self.config_path, is_training=False)

        # Fazer predição
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(processed_data)[0, 1]
        else:
            prediction = float(self.model.predict(processed_data)[0])

        return prediction

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Faz predições para múltiplas amostras

        Args:
            df: DataFrame com os dados

        Returns:
            Array com probabilidades
        """
        # Preprocessar dados
        processed_data = preprocess_data(df, self.config_path, is_training=False)

        # Fazer predições
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(processed_data)[:, 1]
        else:
            predictions = self.model.predict(processed_data)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna importância das features para modelos que suportam

        Returns:
            DataFrame com importância das features
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            features = self.config['preprocessing']['features']

            return pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            logger.warning("Modelo não suporta feature importance")
            return None


def main():
    """Função principal para demonstração"""
    # Configurar paths
    model_path = Path(__file__).parent.parent.parent / 'models' / 'trained' / 'random_forest_latest.joblib'
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'model_config.yaml'

    # Criar preditor
    predictor = ModelPredictor(model_path, config_path)

    # Exemplo de dados para predição
    sample_data = {
        'age': 65,
        'sex': 1,
        'cp': 0,
        'trestbps': 140,
        'chol': 250,
        'fbs': 0,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.5,
        'slope': 1,
        'ca': 0,
        'thal': 2
    }

    # Fazer predição
    prediction = predictor.predict_single(sample_data)
    logger.info(f"Probabilidade de doença cardíaca: {prediction:.2%}")

    # Mostrar importância das features
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        logger.info("\nImportância das Features:")
        print(importance_df)


if __name__ == '__main__':
    main()