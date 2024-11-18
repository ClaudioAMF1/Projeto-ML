import os
import pandas as pd
from pathlib import Path
import yaml
import sys
import logging
from urllib.request import urlretrieve

# Adicionar o diretório raiz ao path do Python
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import setup_logging, load_config

# Configurar logging
logger = logging.getLogger(__name__)
setup_logging()


def download_data(url: str, output_path: Path) -> None:
    """
    Download dos dados brutos do UCI ML Repository

    Args:
        url: URL dos dados
        output_path: Caminho para salvar os dados
    """
    try:
        logger.info(f"Baixando dados de {url}")
        urlretrieve(url, output_path)
        logger.info(f"Dados salvos em {output_path}")
    except Exception as e:
        logger.error(f"Erro ao baixar dados: {str(e)}")
        raise


def create_initial_dataframe(input_path: Path) -> pd.DataFrame:
    """
    Cria DataFrame inicial com os dados brutos

    Args:
        input_path: Caminho para o arquivo de dados brutos

    Returns:
        DataFrame processado
    """
    # Nomes das colunas para o dataset de doenças cardíacas
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    try:
        # Ler dados brutos
        df = pd.read_csv(input_path, names=columns, na_values='?')
        logger.info(f"DataFrame criado com {len(df)} linhas e {len(df.columns)} colunas")
        return df
    except Exception as e:
        logger.error(f"Erro ao criar DataFrame: {str(e)}")
        raise


def perform_initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpeza inicial dos dados

    Args:
        df: DataFrame original

    Returns:
        DataFrame limpo
    """
    try:
        # Criar cópia do DataFrame
        df_clean = df.copy()

        # Converter target para binário (0 = sem doença, 1 = com doença)
        df_clean['target'] = df_clean['target'].apply(lambda x: 1 if x > 0 else 0)

        # Tratar valores ausentes
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        logger.info("Limpeza inicial dos dados concluída")
        return df_clean

    except Exception as e:
        logger.error(f"Erro durante limpeza dos dados: {str(e)}")
        raise


def main():
    """Função principal para preparação dos dados"""

    # Carregar configurações
    config_path = Path(__file__).parents[2] / 'configs' / 'model_config.yaml'
    config = load_config(config_path)

    # Definir caminhos
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = project_dir / 'data' / 'raw' / 'heart_disease_data.csv'
    processed_data_path = project_dir / 'data' / 'processed' / 'heart_disease_processed.csv'

    # Criar diretórios se não existirem
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download dos dados
        download_data(config['data']['url'], raw_data_path)

        # Criar DataFrame inicial
        df = create_initial_dataframe(raw_data_path)

        # Realizar limpeza inicial
        df_clean = perform_initial_cleaning(df)

        # Salvar dados processados
        df_clean.to_csv(processed_data_path, index=False)
        logger.info(f"Dados processados salvos em {processed_data_path}")

    except Exception as e:
        logger.error(f"Erro durante o processamento dos dados: {str(e)}")
        raise


if __name__ == '__main__':
    main()