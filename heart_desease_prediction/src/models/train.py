import logging
import sys
from pathlib import Path
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import tensorflow as tf
from datetime import datetime

# Adicionar o diretório src ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import setup_logging, load_config, save_metrics
from src.data.preprocess import preprocess_data

logger = logging.getLogger(__name__)
setup_logging()


class ModelTrainer:
    """Classe para treinamento de modelos"""

    def __init__(self, config_path: str):
        """
        Inicializa o treinador de modelos

        Args:
            config_path: Caminho para o arquivo de configuração
        """
        self.config = load_config(config_path)
        self.models = {}
        self.setup_paths()

    def setup_paths(self):
        """Configura os caminhos do projeto"""
        self.project_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.project_dir / 'data'
        self.models_dir = self.project_dir / 'models'
        self.reports_dir = self.project_dir / 'reports'

        # Criar diretórios se não existirem
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> tuple:
        """
        Carrega e prepara os dados para treinamento

        Returns:
            Tuple com dados de treino e teste
        """
        logger.info("Carregando dados...")

        # Carregar dados
        data_path = self.data_dir / 'processed' / 'heart_disease_processed.csv'
        df = pd.read_csv(data_path)

        # Separar features e target
        X = df.drop(self.config['data']['target_column'], axis=1)
        y = df[self.config['data']['target_column']]

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_random_forest(self, X_train, y_train):
        """Treina modelo Random Forest"""
        logger.info("Treinando Random Forest...")

        rf_config = self.config['models']['random_forest']
        if rf_config['enabled']:
            rf = RandomForestClassifier(**rf_config['params'])
            rf.fit(X_train, y_train)
            self.models['random_forest'] = rf

    def train_xgboost(self, X_train, y_train):
        """Treina modelo XGBoost"""
        logger.info("Treinando XGBoost...")

        xgb_config = self.config['models']['xgboost']
        if xgb_config['enabled']:
            xgb_model = xgb.XGBClassifier(**xgb_config['params'])
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model

    def train_neural_network(self, X_train, y_train):
        """Treina rede neural"""
        logger.info("Treinando Rede Neural...")

        nn_config = self.config['models']['neural_network']
        if nn_config['enabled']:
            # Criar modelo sequencial
            model = tf.keras.Sequential()

            # Adicionar camadas conforme configuração
            for layer in nn_config['architecture']:
                model.add(tf.keras.layers.Dense(
                    units=layer['units'],
                    activation=layer['activation']
                ))
                if 'dropout' in layer:
                    model.add(tf.keras.layers.Dropout(layer['dropout']))

            # Adicionar camada de saída
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            # Compilar modelo
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Treinar modelo
            history = model.fit(
                X_train, y_train,
                batch_size=nn_config['training']['batch_size'],
                epochs=nn_config['training']['epochs'],
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=nn_config['training']['early_stopping_patience']
                    )
                ]
            )

            self.models['neural_network'] = model
            return history

    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Avalia o modelo e retorna métricas
        """
        logger.info(f"Avaliando modelo {model_name}...")

        # Fazer predições
        if isinstance(model, tf.keras.Model):
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)  # Adicione esta linha
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        # Calcular métricas
        metrics = {}

        # Métricas básicas do classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics.update({
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        })

        # Salvar relatório detalhado
        report_path = self.reports_dir / f"{model_name}_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f)

        return metrics

    def save_models(self):
        """Salva os modelos treinados"""
        logger.info("Salvando modelos...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, model in self.models.items():
            model_path = self.models_dir / 'trained' / f"{name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Modelo {name} salvo em {model_path}")

    def run_training(self):
        """Executa o processo completo de treinamento"""
        logger.info("Iniciando processo de treinamento...")

        try:
            # Carregar e preparar dados
            X_train, X_test, y_train, y_test = self.load_data()

            # Treinar modelos
            self.train_random_forest(X_train, y_train)
            self.train_xgboost(X_train, y_train)
            history = self.train_neural_network(X_train, y_train)

            # Avaliar modelos e salvar métricas
            metrics = {}
            for name, model in self.models.items():
                model_metrics = self.evaluate_model(model, X_test, y_test, name)
                metrics[name] = model_metrics

                logger.info(f"\nMétricas para {name}:")
                for metric_name, value in model_metrics.items():
                    logger.info(f"{metric_name}: {value:.4f}")

            # Salvar métricas gerais
            metrics_path = self.reports_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            with open(metrics_path, 'w') as f:
                yaml.dump(metrics, f)

            # Salvar modelos
            self.save_models()

            logger.info("Treinamento concluído com sucesso!")

        except Exception as e:
            logger.error(f"Erro durante o treinamento: {str(e)}")
            raise


if __name__ == '__main__':
    # Configurar paths
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'model_config.yaml'

    # Criar e executar trainer
    trainer = ModelTrainer(config_path)
    trainer.run_training()