import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações de estilo
plt.style.use('default')
sns.set_theme()
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


class ModelVisualizer:
    def __init__(self):
        """Inicializa o visualizador"""
        self.output_dir = Path(__file__).parent.parent.parent / 'reports' / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório de saída: {self.output_dir}")

    def save_plot(self, name):
        """Salva o plot atual"""
        plt.tight_layout()
        output_path = self.output_dir / f"{name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot salvo em {output_path}")
        plt.close()

    def plot_feature_distributions(self, df, numerical_features, categorical_features):
        """Plota distribuições das features numéricas e categóricas"""
        # Features numéricas
        n_num = len(numerical_features)
        if n_num > 0:
            fig, axes = plt.subplots(2, (n_num + 1) // 2, figsize=(15, 10))
            axes = axes.ravel()

            for idx, feature in enumerate(numerical_features):
                sns.histplot(data=df, x=feature, ax=axes[idx], kde=True)
                axes[idx].set_title(f'Distribuição de {feature}')

            plt.tight_layout()
            self.save_plot('numerical_distributions')

        # Features categóricas
        n_cat = len(categorical_features)
        if n_cat > 0:
            fig, axes = plt.subplots(2, (n_cat + 1) // 2, figsize=(15, 10))
            axes = axes.ravel()

            for idx, feature in enumerate(categorical_features):
                sns.countplot(data=df, x=feature, ax=axes[idx])
                axes[idx].set_title(f'Contagem de {feature}')
                axes[idx].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            self.save_plot('categorical_distributions')

    def plot_correlation_matrix(self, df_numerical):
        """Plota matriz de correlação para features numéricas"""
        plt.figure(figsize=(10, 8))
        corr_matrix = df_numerical.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação')
        self.save_plot('correlation_matrix')

    def plot_target_relations(self, df, numerical_features, categorical_features, target):
        """Plota relações entre features e target"""
        # Features numéricas vs target
        n_num = len(numerical_features)
        if n_num > 0:
            fig, axes = plt.subplots(2, (n_num + 1) // 2, figsize=(15, 10))
            axes = axes.ravel()

            for idx, feature in enumerate(numerical_features):
                sns.boxplot(data=df, x=target, y=feature, ax=axes[idx])
                axes[idx].set_title(f'{feature} vs {target}')

            plt.tight_layout()
            self.save_plot('numerical_vs_target')

        # Features categóricas vs target
        n_cat = len(categorical_features)
        if n_cat > 0:
            fig, axes = plt.subplots(2, (n_cat + 1) // 2, figsize=(15, 10))
            axes = axes.ravel()

            for idx, feature in enumerate(categorical_features):
                sns.barplot(data=df, x=feature, y=target, ax=axes[idx])
                axes[idx].set_title(f'{feature} vs {target}')
                axes[idx].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            self.save_plot('categorical_vs_target')

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plota matriz de confusão"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        self.save_plot('confusion_matrix')

    def plot_roc_curve(self, y_true, y_pred_proba, title="ROC Curve"):
        """Plota curva ROC"""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        self.save_plot('roc_curve')

    def plot_feature_importance(self, feature_importance_df, title="Feature Importance"):
        """Plota importância das features"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        self.save_plot('feature_importance')


def main():
    """Função principal"""
    try:
        # Definir caminhos
        project_dir = Path(__file__).parent.parent.parent
        data_path = project_dir / 'data' / 'processed' / 'heart_disease_processed.csv'
        models_dir = project_dir / 'models' / 'trained'

        # Verificar se o arquivo de dados existe
        if not data_path.exists():
            logger.error(f"Arquivo de dados não encontrado: {data_path}")
            return

        # Carregar dados
        logger.info("Carregando dados...")
        df = pd.read_csv(data_path)

        # Definir features
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

        # Criar visualizador
        visualizer = ModelVisualizer()

        # 1. Plotar distribuições
        logger.info("Gerando visualizações de distribuições...")
        visualizer.plot_feature_distributions(df, numerical_features, categorical_features)

        # 2. Plotar matriz de correlação
        logger.info("Gerando matriz de correlação...")
        visualizer.plot_correlation_matrix(df[numerical_features])

        # 3. Plotar relações com target
        logger.info("Gerando visualizações de relações com target...")
        visualizer.plot_target_relations(df, numerical_features, categorical_features, 'target')

        # 4. Carregar o melhor modelo (Random Forest)
        rf_models = list(models_dir.glob('random_forest*.joblib'))
        if rf_models:
            latest_model = sorted(rf_models)[-1]
            logger.info(f"Carregando modelo: {latest_model}")

            model = joblib.load(latest_model)

            # Preparar dados para predição
            X = df.drop('target', axis=1)
            y = df['target']

            # Fazer predições
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            # 5. Plotar matriz de confusão
            logger.info("Gerando matriz de confusão...")
            visualizer.plot_confusion_matrix(y, y_pred, "Random Forest - Confusion Matrix")

            # 6. Plotar curva ROC
            logger.info("Gerando curva ROC...")
            visualizer.plot_roc_curve(y, y_pred_proba, "Random Forest - ROC Curve")

            # 7. Plotar importância das features
            if hasattr(model, 'feature_importances_'):
                logger.info("Gerando gráfico de importância das features...")
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                visualizer.plot_feature_importance(importance_df, "Random Forest - Feature Importance")

        logger.info("Visualizações concluídas com sucesso!")

    except Exception as e:
        logger.error(f"Erro durante a visualização: {str(e)}")
        raise


if __name__ == '__main__':
    main()