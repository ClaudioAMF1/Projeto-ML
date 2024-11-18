import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.web.utils.utils import export_to_excel, create_download_link


class DataAnalyzer:
    def __init__(self):
        self.load_data()

    def load_data(self):
        """Carrega dados processados"""
        data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'heart_disease_processed.csv'
        self.df = pd.read_csv(data_path)

    def show_basic_stats(self):
        """Mostra estatísticas básicas"""
        st.subheader("Estatísticas Básicas")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Pacientes", len(self.df))

        with col2:
            positive_cases = (self.df['target'] == 1).sum()
            st.metric("Casos Positivos", positive_cases,
                      f"{positive_cases / len(self.df):.1%}")

        with col3:
            negative_cases = (self.df['target'] == 0).sum()
            st.metric("Casos Negativos", negative_cases,
                      f"{negative_cases / len(self.df):.1%}")

    def plot_age_distribution(self):
        """Plota distribuição de idade"""
        fig = px.histogram(self.df, x='age', color='target',
                           title='Distribuição de Idade por Diagnóstico',
                           labels={'target': 'Doença Cardíaca'})
        st.plotly_chart(fig)

    def plot_correlation_matrix(self):
        """Plota matriz de correlação"""
        corr = self.df.corr()
        fig = px.imshow(corr, title='Matriz de Correlação',
                        labels=dict(color="Correlação"))
        st.plotly_chart(fig)

    def plot_feature_distributions(self, feature):
        """Plota distribuição de uma feature específica"""
        if self.df[feature].dtype in ['int64', 'float64']:
            fig = px.box(self.df, x='target', y=feature,
                         title=f'Distribuição de {feature} por Diagnóstico')
        else:
            fig = px.histogram(self.df, x=feature, color='target',
                               title=f'Distribuição de {feature} por Diagnóstico')
        st.plotly_chart(fig)

    def generate_summary_report(self):
        """Gera relatório resumido"""
        summary = {
            'total_patients': len(self.df),
            'positive_cases': (self.df['target'] == 1).sum(),
            'negative_cases': (self.df['target'] == 0).sum(),
            'age_mean': self.df['age'].mean(),
            'age_std': self.df['age'].std(),
            'male_ratio': (self.df['sex'] == 1).mean(),
            'female_ratio': (self.df['sex'] == 0).mean()
        }

        # Criar DataFrame para exportação
        summary_df = pd.DataFrame([summary])

        # Exportar para Excel
        output_path = export_to_excel(summary_df, 'summary_report.xlsx')

        # Criar link de download
        st.markdown(create_download_link(output_path, 'Download Relatório Resumido'),
                    unsafe_allow_html=True)

    def plot_risk_factors(self):
        """Plota fatores de risco"""
        risk_factors = ['age', 'trestbps', 'chol', 'thalach']

        for factor in risk_factors:
            fig = go.Figure()

            for target in [0, 1]:
                fig.add_trace(go.Violin(
                    x=self.df[self.df['target'] == target][factor],
                    name=f"{'Com' if target == 1 else 'Sem'} Doença",
                    box_visible=True,
                    meanline_visible=True
                ))

            fig.update_layout(title=f'Distribuição de {factor} por Diagnóstico')
            st.plotly_chart(fig)

    def show_analysis(self):
        """Mostra análise completa"""
        st.title("📊 Análise de Dados")

        # Estatísticas básicas
        self.show_basic_stats()

        # Tabs para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["Distribuições", "Correlações", "Fatores de Risco"])

        with tab1:
            st.subheader("Distribuições")
            self.plot_age_distribution()

            # Seletor de feature
            feature = st.selectbox(
                "Escolha uma característica para análise:",
                self.df.columns.drop('target')
            )
            self.plot_feature_distributions(feature)

        with tab2:
            st.subheader("Correlações")
            self.plot_correlation_matrix()

        with tab3:
            st.subheader("Fatores de Risco")
            self.plot_risk_factors()

        # Gerar relatório
        st.subheader("Relatório")
        if st.button("Gerar Relatório Resumido"):
            self.generate_summary_report()