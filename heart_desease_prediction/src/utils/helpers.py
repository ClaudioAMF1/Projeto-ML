import logging
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List


def setup_logging(log_level=logging.INFO) -> None:
    """Configura o logging básico do projeto"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carrega arquivo de configuração YAML

    Args:
        config_path: Caminho para o arquivo de configuração

    Returns:
        Dict com as configurações
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def save_metrics(metrics: Dict[str, float], output_path: Union[str, Path]) -> None:
    """
    Salva métricas de avaliação em arquivo CSV

    Args:
        metrics: Dicionário com métricas
        output_path: Caminho para salvar o arquivo
    """
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)


def create_experiment_folder(base_path: Union[str, Path], experiment_name: str) -> Path:
    """
    Cria pasta para um novo experimento

    Args:
        base_path: Caminho base para experimentos
        experiment_name: Nome do experimento

    Returns:
        Path da pasta criada
    """
    experiment_path = Path(base_path) / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    return experiment_path


class DataFrameInfo:
    """Classe para análise básica de DataFrames"""

    @staticmethod
    def get_missing_values(df: pd.DataFrame) -> pd.Series:
        """Retorna informações sobre valores ausentes"""
        return df.isnull().sum()

    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Retorna estatísticas básicas do DataFrame"""
        return df.describe()

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Retorna matriz de correlação"""
        return df.corr()


def timer_decorator(func):
    """Decorator para medir tempo de execução de funções"""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Função {func.__name__} executada em {end_time - start_time:.2f} segundos")
        return result

    return wrapper