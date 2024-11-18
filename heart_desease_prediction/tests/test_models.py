import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Adicionar diretório src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.train import ModelTrainer
from src.models.predict import ModelPredictor
from src.data.preprocess import preprocess_data


@pytest.fixture
def sample_data():
    """Fixture para criar dados de teste"""
    return pd.DataFrame({
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [3, 2, 0, 2, 1],
        'trestbps': [145, 160, 120, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [0, 0, 0, 1, 0],
        'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
        'slope': [0, 1, 1, 0, 2],
        'ca': [0, 3, 2, 0, 0],
        'thal': [1, 2, 2, 2, 2],
        'target': [1, 1, 1, 0, 0]
    })


@pytest.fixture
def config_path():
    """Fixture para caminho do arquivo de configuração"""
    return Path(__file__).parent.parent / 'configs' / 'model_config.yaml'


def test_model_trainer_initialization(config_path):
    """Testa inicialização do ModelTrainer"""
    trainer = ModelTrainer(config_path)
    assert trainer is not None
    assert hasattr(trainer, 'config')
    assert hasattr(trainer, 'models')


def test_data_preprocessing(sample_data, config_path):
    """Testa preprocessamento dos dados"""
    # Separar features e target
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']

    # Preprocessar dados
    processed_data = preprocess_data(X, config_path)

    assert processed_data is not None
    assert len(processed_data) == len(sample_data)
    assert not np.isnan(processed_data).any()


def test_model_training(sample_data, config_path):
    """Testa treinamento do modelo"""
    trainer = ModelTrainer(config_path)

    # Separar features e target
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']

    # Treinar modelo Random Forest
    trainer.train_random_forest(X, y)

    assert 'random_forest' in trainer.models
    assert trainer.models['random_forest'] is not None


def test_model_prediction(sample_data, config_path):
    """Testa predições do modelo"""
    # Treinar modelo
    trainer = ModelTrainer(config_path)
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    trainer.train_random_forest(X, y)

    # Salvar modelo temporariamente
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        from joblib import dump
        dump(trainer.models['random_forest'], tmp.name)

        # Criar preditor
        predictor = ModelPredictor(tmp.name, config_path)

        # Fazer predições
        predictions = predictor.predict_batch(X)

        assert len(predictions) == len(sample_data)
        assert all(isinstance(pred, (np.float64, float)) for pred in predictions)
        assert all(0 <= pred <= 1 for pred in predictions)


def test_feature_importance(sample_data, config_path):
    """Testa cálculo de importância das features"""
    trainer = ModelTrainer(config_path)
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    trainer.train_random_forest(X, y)

    # Salvar modelo temporariamente
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        from joblib import dump
        dump(trainer.models['random_forest'], tmp.name)

        # Criar preditor
        predictor = ModelPredictor(tmp.name, config_path)

        # Calcular importância das features
        importance_df = predictor.get_feature_importance()

        assert importance_df is not None
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) > 0


def test_error_handling(config_path):
    """Testa tratamento de erros"""
    with pytest.raises(Exception):
        # Tentar carregar dados inexistentes
        trainer = ModelTrainer(config_path)
        trainer.load_data()


if __name__ == '__main__':
    pytest.main([__file__])