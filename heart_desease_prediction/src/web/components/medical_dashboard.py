import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np


class MedicalDashboard:
    def __init__(self):
        self.load_data()

    def load_data(self):
        """Carrega dados necessários para o dashboard"""
        # Carregar dados dos pacientes
        patients_path = Path(__file__).parent.parent / 'assets' / 'patient_history.json'
        if patients_path.exists():
            with open(patients_path, 'r') as f:
                self.patient_data = json.load(f)
        else:
            self.patient_data = {}

        # Carregar dados processados
        data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'heart_disease_processed.csv'
        self.df = pd.read_csv(data_path)

    def get_summary_metrics(self):
        """Calcula métricas resumidas"""
        total_patients = len(self.patient_data)

        # Calcular pacientes de alto risco
        high_risk_count = sum(
            1 for patient in self.patient_data.values()
            if patient['predictions'] and patient['predictions'][-1]['prediction'] > 0.7
        )

        # Calcular consultas nos últimos 30 dias
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_predictions = sum(
            1 for patient in self.patient_data.values()
            for pred in patient['predictions']
            if datetime.strptime(pred['timestamp'], '%Y-%m-%d %H:%M:%S') > thirty_days_ago
        )

        return {
            'total_patients': total_patients,
            'high_risk': high_risk_count,
            'recent_predictions': recent_predictions
        }

    def plot_risk_distribution(self):
        """Plota distribuição de risco dos pacientes"""
        risk_values = [
            patient['predictions'][-1]['prediction']
            for patient in self.patient_data.values()
            if patient['predictions']
        ]

        if not risk_values:
            return None

        fig = px.histogram(
            x=risk_values,
            nbins=20,
            title='Distribuição de Risco Cardíaco',
            labels={'x': 'Risco', 'y': 'Número de Pacientes'},
            color_discrete_sequence=['#FF4B4B']
        )

        return fig

    def plot_risk_by_age(self):
        """Plota relação entre risco e idade"""
        data = []
        for patient in self.patient_data.values():
            if patient['predictions']:
                birth_date = datetime.strptime(patient['birth_date'], '%Y-%m-%d')
                age = (datetime.now() - birth_date).days / 365.25
                risk = patient['predictions'][-1]['prediction']
                data.append({'age': age, 'risk': risk})

        if not data:
            return None

        df = pd.DataFrame(data)
        fig = px.scatter(
            df,
            x='age',
            y='risk',
            title='Risco Cardíaco por Idade',
            labels={'age': 'Idade', 'risk': 'Risco'},
            trendline="lowess"
        )

        return fig

    def plot_predictions_timeline(self):
        """Plota linha do tempo de predições"""
        data = []
        for patient_id, patient in self.patient_data.items():
            for pred in patient['predictions']:
                data.append({
                    'timestamp': datetime.strptime(pred['timestamp'], '%Y-%m-%d %H:%M:%S'),
                    'prediction': pred['prediction'],
                    'patient_name': patient['name']
                })

        if not data:
            return None

        df = pd.DataFrame(data)
        fig = px.line(
            df,
            x='timestamp',
            y='prediction',
            color='patient_name',
            title='Evolução do Risco ao Longo do Tempo'
        )

        return fig

    def create_high_risk_table(self):
        """Cria tabela de pacientes de alto risco"""
        high_risk_patients = []

        for patient_id, patient in self.patient_data.items():
            if patient['predictions'] and patient['predictions'][-1]['prediction'] > 0.7:
                high_risk_patients.append({
                    'ID': patient_id,
                    'Nome': patient['name'],
                    'Risco': f"{patient['predictions'][-1]['prediction']:.1%}",
                    'Última Avaliação': patient['predictions'][-1]['timestamp']
                })

        return pd.DataFrame(high_risk_patients)

    def show_dashboard(self):
        """Mostra dashboard completo"""
        st.title("📊 Dashboard Médico")

        # Métricas resumidas
        metrics = self.get_summary_metrics()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Pacientes", metrics['total_patients'])

        with col2:
            st.metric("Pacientes de Alto Risco",
                      metrics['high_risk'],
                      f"{metrics['high_risk'] / metrics['total_patients']:.1%}" if metrics[
                                                                                       'total_patients'] > 0 else "0%")

        with col3:
            st.metric("Avaliações (30 dias)", metrics['recent_predictions'])

        # Tabs para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["Distribuição de Risco", "Análise por Idade", "Evolução Temporal"])

        with tab1:
            fig = self.plot_risk_distribution()
            if fig:
                st.plotly_chart(fig)
            else:
                st.info("Sem dados suficientes para visualização")

        with tab2:
            fig = self.plot_risk_by_age()
            if fig:
                st.plotly_chart(fig)
            else:
                st.info("Sem dados suficientes para visualização")

        with tab3:
            fig = self.plot_predictions_timeline()
            if fig:
                st.plotly_chart(fig)
            else:
                st.info("Sem dados suficientes para visualização")

        # Lista de pacientes de alto risco
        st.subheader("⚠️ Pacientes de Alto Risco")
        high_risk_df = self.create_high_risk_table()
        if not high_risk_df.empty:
            st.dataframe(high_risk_df)
        else:
            st.info("Nenhum paciente de alto risco identificado")

        # Exportar dados
        if st.button("📥 Exportar Relatório"):
            output_path = Path(__file__).parent.parent / 'assets' / 'medical_report.xlsx'

            with pd.ExcelWriter(output_path) as writer:
                # Métricas gerais
                pd.DataFrame([metrics]).to_excel(writer, sheet_name='Métricas Gerais', index=False)

                # Pacientes de alto risco
                if not high_risk_df.empty:
                    high_risk_df.to_excel(writer, sheet_name='Pacientes de Alto Risco', index=False)

            st.success(f"Relatório exportado para {output_path}")

    def plot_demographic_analysis(self):
        """Realiza análise demográfica dos pacientes"""
        if not self.patient_data:
            return None

        # Análise por gênero
        gender_data = pd.DataFrame([
            {'gender': patient['gender']}
            for patient in self.patient_data.values()
        ])

        fig1 = px.pie(
            gender_data,
            names='gender',
            title='Distribuição por Gênero'
        )

        # Análise por faixa etária
        age_data = []
        for patient in self.patient_data.values():
            birth_date = datetime.strptime(patient['birth_date'], '%Y-%m-%d')
            age = (datetime.now() - birth_date).days / 365.25

            if age < 30:
                age_group = '<30'
            elif age < 45:
                age_group = '30-45'
            elif age < 60:
                age_group = '45-60'
            else:
                age_group = '>60'

            age_data.append({'age_group': age_group})

        age_df = pd.DataFrame(age_data)
        fig2 = px.bar(
            age_df['age_group'].value_counts().reset_index(),
            x='index',
            y='age_group',
            title='Distribuição por Faixa Etária'
        )

        return fig1, fig2

    def plot_risk_trends(self):
        """Analisa tendências de risco ao longo do tempo"""
        if not self.patient_data:
            return None

        # Média móvel do risco
        data = []
        for patient in self.patient_data.values():
            for pred in patient['predictions']:
                data.append({
                    'timestamp': datetime.strptime(pred['timestamp'], '%Y-%m-%d %H:%M:%S'),
                    'prediction': pred['prediction']
                })

        if not data:
            return None

        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        df['rolling_mean'] = df['prediction'].rolling(window=7).mean()

        fig = px.line(
            df,
            x='timestamp',
            y=['prediction', 'rolling_mean'],
            title='Tendência de Risco (Média Móvel 7 dias)'
        )

        return fig