import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import base64
from fpdf import FPDF
import json
from typing import Dict, Any, List
import io


class ReportGenerator:
    def __init__(self):
        """Inicializa o gerador de relatórios"""
        self.report_dir = Path(__file__).parent.parent / 'assets' / 'reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_patient_report(self, patient_data: Dict[str, Any]) -> str:
        """
        Gera relatório detalhado do paciente em PDF

        Args:
            patient_data: Dicionário com dados do paciente

        Returns:
            Caminho do arquivo PDF gerado
        """
        pdf = FPDF()
        pdf.add_page()

        # Cabeçalho
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Relatório de Avaliação Cardíaca', ln=True, align='C')
        pdf.ln(10)

        # Informações do Paciente
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Informações do Paciente', ln=True)
        pdf.set_font('Arial', '', 12)

        info_list = [
            f"Nome: {patient_data['name']}",
            f"Data de Nascimento: {patient_data['birth_date']}",
            f"Gênero: {patient_data['gender']}",
            f"Email: {patient_data['email']}",
            f"Telefone: {patient_data['phone']}"
        ]

        for info in info_list:
            pdf.cell(0, 8, info, ln=True)

        # Histórico Médico
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Histórico Médico', ln=True)
        pdf.set_font('Arial', '', 12)

        med_info = [
            f"Histórico Familiar: {', '.join(patient_data['family_history'])}",
            f"Status Tabagismo: {patient_data['smoking']}",
            f"Medicamentos: {patient_data['medications']}",
            f"Alergias: {patient_data['allergies']}"
        ]

        for info in med_info:
            pdf.multi_cell(0, 8, info)

        # Histórico de Avaliações
        if patient_data['predictions']:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Histórico de Avaliações', ln=True)
            pdf.set_font('Arial', '', 12)

            for idx, pred in enumerate(patient_data['predictions'], 1):
                pdf.multi_cell(0, 8, (
                    f"Avaliação {idx}:\n"
                    f"Data: {pred['timestamp']}\n"
                    f"Risco: {pred['prediction']:.1%}\n"
                    "---"
                ))

        # Gerar nome do arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"patient_report_{timestamp}.pdf"
        filepath = self.report_dir / filename

        # Salvar PDF
        pdf.output(str(filepath))
        return filepath

    def generate_medical_summary(self, data: Dict[str, Any]) -> str:
        """
        Gera relatório resumido médico em Excel

        Args:
            data: Dicionário com dados para o relatório

        Returns:
            Caminho do arquivo Excel gerado
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"medical_summary_{timestamp}.xlsx"
        filepath = self.report_dir / filename

        with pd.ExcelWriter(filepath) as writer:
            # Métricas Gerais
            pd.DataFrame([data['metrics']]).to_excel(
                writer, sheet_name='Métricas Gerais', index=False
            )

            # Pacientes de Alto Risco
            if data.get('high_risk_patients'):
                pd.DataFrame(data['high_risk_patients']).to_excel(
                    writer, sheet_name='Pacientes de Alto Risco', index=False
                )

            # Estatísticas
            if data.get('statistics'):
                pd.DataFrame(data['statistics']).to_excel(
                    writer, sheet_name='Estatísticas', index=False
                )

        return filepath

    def generate_risk_analysis_report(self, predictions: List[Dict[str, Any]]) -> str:
        """
        Gera relatório de análise de risco em PDF

        Args:
            predictions: Lista de predições com metadados

        Returns:
            Caminho do arquivo PDF gerado
        """
        pdf = FPDF()
        pdf.add_page()

        # Cabeçalho
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Análise de Risco Cardíaco', ln=True, align='C')
        pdf.ln(10)

        # Estatísticas
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Estatísticas Gerais', ln=True)
        pdf.set_font('Arial', '', 12)

        # Calcular estatísticas
        risks = [pred['prediction'] for pred in predictions]
        stats = {
            'Número de Avaliações': len(risks),
            'Risco Médio': sum(risks) / len(risks),
            'Risco Máximo': max(risks),
            'Risco Mínimo': min(risks)
        }

        for key, value in stats.items():
            if isinstance(value, float):
                pdf.cell(0, 8, f"{key}: {value:.1%}", ln=True)
            else:
                pdf.cell(0, 8, f"{key}: {value}", ln=True)

        # Distribuição de Risco
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Distribuição de Risco', ln=True)
        pdf.set_font('Arial', '', 12)

        risk_levels = {
            'Alto Risco (>70%)': sum(1 for r in risks if r > 0.7),
            'Risco Médio (30-70%)': sum(1 for r in risks if 0.3 < r <= 0.7),
            'Baixo Risco (<30%)': sum(1 for r in risks if r <= 0.3)
        }

        for level, count in risk_levels.items():
            percentage = (count / len(risks)) * 100
            pdf.cell(0, 8, f"{level}: {count} ({percentage:.1f}%)", ln=True)

        # Gerar nome do arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"risk_analysis_{timestamp}.pdf"
        filepath = self.report_dir / filename

        # Salvar PDF
        pdf.output(str(filepath))
        return filepath

    def get_download_link(self, file_path: Path, text: str) -> str:
        """
        Cria link de download para arquivo

        Args:
            file_path: Caminho do arquivo
            text: Texto do link

        Returns:
            HTML do link de download
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path.name}">{text}</a>'

    def show_report_options(self):
        """Mostra interface para geração de relatórios"""
        st.title("📊 Gerador de Relatórios")

        report_type = st.selectbox(
            "Selecione o tipo de relatório:",
            ["Relatório de Paciente", "Resumo Médico", "Análise de Risco"]
        )

        if report_type == "Relatório de Paciente":
            # Carregar dados dos pacientes
            patient_file = Path(__file__).parent.parent / 'assets' / 'patient_history.json'
            if patient_file.exists():
                with open(patient_file, 'r') as f:
                    patients = json.load(f)

                patient_id = st.selectbox(
                    "Selecione o paciente:",
                    options=list(patients.keys()),
                    format_func=lambda x: f"{patients[x]['name']} (ID: {x})"
                )

                if st.button("Gerar Relatório"):
                    with st.spinner("Gerando relatório..."):
                        filepath = self.generate_patient_report(patients[patient_id])
                        st.success("Relatório gerado com sucesso!")
                        st.markdown(
                            self.get_download_link(filepath, "📥 Download Relatório"),
                            unsafe_allow_html=True
                        )

        elif report_type == "Resumo Médico":
            data = {
                'metrics': {
                    'total_patients': 100,
                    'high_risk': 20,
                    'medium_risk': 45,
                    'low_risk': 35
                },
                'high_risk_patients': [
                    {'id': 1, 'name': 'Paciente A', 'risk': 0.85},
                    {'id': 2, 'name': 'Paciente B', 'risk': 0.78}
                ]
            }

            if st.button("Gerar Relatório"):
                with st.spinner("Gerando relatório..."):
                    filepath = self.generate_medical_summary(data)
                    st.success("Relatório gerado com sucesso!")
                    st.markdown(
                        self.get_download_link(filepath, "📥 Download Relatório"),
                        unsafe_allow_html=True
                    )

        else:  # Análise de Risco
            # Carregar dados de predições
            predictions = [
                {'prediction': 0.85, 'timestamp': '2024-01-01'},
                {'prediction': 0.45, 'timestamp': '2024-01-02'},
                {'prediction': 0.25, 'timestamp': '2024-01-03'}
            ]

            if st.button("Gerar Relatório"):
                with st.spinner("Gerando relatório..."):
                    filepath = self.generate_risk_analysis_report(predictions)
                    st.success("Relatório gerado com sucesso!")
                    st.markdown(
                        self.get_download_link(filepath, "📥 Download Relatório"),
                        unsafe_allow_html=True
                    )