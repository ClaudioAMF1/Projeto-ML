import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class CustomMissingImputer(BaseEstimator, TransformerMixin):
    """
    Imputer personalizado para tratamento de valores ausentes
    """

    def __init__(self, numerical_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent'):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.numerical_fill_values = {}
        self.categorical_fill_values = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # Para colunas numéricas
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if self.numerical_strategy == 'median':
                self.numerical_fill_values[col] = X[col].median()
            elif self.numerical_strategy == 'mean':
                self.numerical_fill_values[col] = X[col].mean()

        # Para colunas categóricas
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.categorical_strategy == 'most_frequent':
                self.categorical_fill_values[col] = X[col].mode()[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Preenchendo valores ausentes
        for col, value in self.numerical_fill_values.items():
            X[col] = X[col].fillna(value)

        for col, value in self.categorical_fill_values.items():
            X[col] = X[col].fillna(value)

        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Classe para criação de novas features
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Criando novas features
        if 'age' in X.columns and 'thalach' in X.columns:
            X['age_thalach_ratio'] = X['age'] / X['thalach']

        if 'oldpeak' in X.columns:
            X['oldpeak_squared'] = X['oldpeak'] ** 2

        if 'chol' in X.columns:
            X['chol_level'] = pd.cut(X['chol'],
                                     bins=[0, 200, 240, float('inf')],
                                     labels=['normal', 'borderline', 'high'])

        return X


def create_preprocessing_pipeline(
        numerical_features: List[str],
        categorical_features: List[str]
) -> Pipeline:
    """
    Cria pipeline de preprocessamento completo

    Args:
        numerical_features: Lista de features numéricas
        categorical_features: Lista de features categóricas

    Returns:
        Pipeline de preprocessamento
    """

    # Pipeline para features numéricas
    numerical_pipeline = Pipeline([
        ('imputer', CustomMissingImputer(numerical_strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para features categóricas
    categorical_pipeline = Pipeline([
        ('imputer', CustomMissingImputer(categorical_strategy='most_frequent')),
        ('encoder', LabelEncoder())
    ])

    # Combinando os pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    # Pipeline final com feature engineering
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_engineer', FeatureEngineer())
    ])

    return final_pipeline


def preprocess_data(
        df: pd.DataFrame,
        config_path: str,
        is_training: bool = True
) -> pd.DataFrame:
    """
    Função principal para preprocessamento dos dados

    Args:
        df: DataFrame com os dados
        config_path: Caminho para arquivo de configuração
        is_training: Indica se é processamento de treino ou teste

    Returns:
        DataFrame processado
    """
    try:
        # Carregando configurações
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Criando pipeline
        pipeline = create_preprocessing_pipeline(
            numerical_features=config['preprocessing']['numerical_features'],
            categorical_features=config['preprocessing']['categorical_features']
        )

        # Aplicando preprocessamento
        if is_training:
            processed_data = pipeline.fit_transform(df)
        else:
            processed_data = pipeline.transform(df)

        logger.info("Preprocessamento concluído com sucesso")
        return processed_data

    except Exception as e:
        logger.error(f"Erro durante preprocessamento: {str(e)}")
        raise