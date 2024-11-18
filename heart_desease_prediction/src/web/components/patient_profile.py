import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import base64


class PatientProfile:
    def __init__(self):
        self.load_patient_data()

    def load_patient_data(self):
        """Carrega histórico de dados dos pacientes"""
        data_path = Path(__file__).parent.parent / 'assets' / 'patient_history.json'
        if data_path.exists():
            with open(data_path, 'r') as f:
                self.patient_data = json.load(f)
        else:
            self.patient_data = {}

    def save_patient_data(self):
        """Salva dados dos pacientes"""
        data_path = Path(__file__).parent.parent / 'assets' / 'patient_history.json'
        with open(data_path, 'w') as f:
            json.dump(self.patient_data, f)

    def add_patient(self, patient_info: dict):
        """Adiciona novo paciente"""
        patient_id = str(len(self.patient_data) + 1)
        patient_info['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        patient_info['predictions'] = []
        self.patient_data[patient_id] = patient_info
        self.save_patient_data()
        return patient_id

    def update_patient(self, patient_id: str, prediction: float, features: dict):
        """Atualiza dados do paciente com nova predição"""
        if patient_id in self.patient_data:
            prediction_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': prediction,
                'features': features
            }
            self.patient_data[patient_id]['predictions'].append(prediction_data)
            self.save_patient_data()

    def show_patient_form(self):
        """Mostra formulário de cadastro de paciente"""
        st.subheader("📋 Cadastro de Paciente")

        with st.form("patient_form"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Nome completo")
                birth_date = st.date_input("Data de nascimento")
                gender = st.selectbox("Gênero", ["Masculino", "Feminino", "Outro"])

            with col2:
                email = st.text_input("Email")
                phone = st.text_input("Telefone")
                address = st.text_area("Endereço")

            # Histórico médico
            st.subheader("Histórico Médico")
            col3, col4 = st.columns(2)

            with col3:
                family_history = st.multiselect(
                    "Histórico familiar de doenças cardíacas",
                    ["Pai", "Mãe", "Irmãos", "Avós"]
                )
                smoking = st.selectbox("Fumante?", ["Não", "Sim", "Ex-fumante"])

            with col4:
                medications = st.text_area("Medicamentos em uso")
                allergies = st.text_area("Alergias")

            submitted = st.form_submit_button("Cadastrar Paciente")

            if submitted:
                patient_info = {
                    'name': name,
                    'birth_date': str(birth_date),
                    'gender': gender,
                    'email': email,
                    'phone': phone,
                    'address': address,
                    'family_history': family_history,
                    'smoking': smoking,
                    'medications': medications,
                    'allergies': allergies
                }

                patient_id = self.add_patient(patient_info)
                st.success(f"Paciente cadastrado com sucesso! ID: {patient_id}")
                return patient_id

        return None

    def show_patient_history(self, patient_id: str):
        """Mostra histórico do paciente"""
        if patient_id not in self.patient_data:
            st.error("Paciente não encontrado!")
            return

        patient = self.patient_data[patient_id]

        # Informações básicas
        st.subheader("🧑‍⚕️ Informações do Paciente")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Nome", patient['name'])

        with col2:
            age = datetime.now().year - datetime.strptime(patient['birth_date'], '%Y-%m-%d').year
            st.metric("Idade", f"{age} anos")

        with col3:
            st.metric("Gênero", patient['gender'])

        # Histórico médico
        st.subheader("📋 Histórico Médico")
        col4, col5 = st.columns(2)

        with col4:
            st.write("**Histórico Familiar:**")
            for relative in patient['family_history']:
                st.write(f"- {relative}")

        with col5:
            st.write("**Status Tabagismo:**", patient['smoking'])

        # Histórico de predições
        if patient['predictions']:
            st.subheader("📊 Histórico de Avaliações")

            # Criar DataFrame com histórico
            history_df = pd.DataFrame([
                {
                    'data': pred['timestamp'],
                    'risco': pred['prediction']
                }
                for pred in patient['predictions']
            ])

            # Plotar evolução do risco
            fig = px.line(history_df, x='data', y='risco',
                          title='Evolução do Risco Cardíaco')
            st.plotly_chart(fig)

            # Tabela com histórico detalhado
            st.write("Histórico Detalhado:")
            st.dataframe(history_df)

        # Exportar dados
        if st.button("Exportar Dados do Paciente"):
            # Criar relatório em Excel
            output_path = Path(__file__).parent.parent / 'assets' / f'patient_{patient_id}_report.xlsx'
            with pd.ExcelWriter(output_path) as writer:
                # Informações básicas
                pd.DataFrame([{k: v for k, v in patient.items() if k != 'predictions'}]).to_excel(
                    writer, sheet_name='Informações Básicas', index=False
                )

                # Histórico de predições
                if patient['predictions']:
                    pd.DataFrame(patient['predictions']).to_excel(
                        writer, sheet_name='Histórico de Predições', index=False
                    )

            # Criar link de download
            with open(output_path, 'rb') as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'data:application/octet-stream;base64,{b64}'

            st.markdown(
                f'<a href="{href}" download="{output_path.name}">Download Relatório do Paciente</a>',
                unsafe_allow_html=True
            )

    def show_search_patient(self):
        """Mostra formulário de busca de paciente"""
        st.subheader("🔍 Buscar Paciente")

        # Opções de busca
        search_option = st.radio(
            "Buscar por:",
            ["ID", "Nome", "Email"]
        )

        search_term = st.text_input("Digite o termo de busca:")

        if search_term:
            found_patients = []

            for pid, patient in self.patient_data.items():
                if search_option == "ID" and pid == search_term:
                    found_patients.append((pid, patient))
                elif search_option == "Nome" and search_term.lower() in patient['name'].lower():
                    found_patients.append((pid, patient))
                elif search_option == "Email" and search_term.lower() in patient['email'].lower():
                    found_patients.append((pid, patient))

            if found_patients:
                st.write(f"Encontrados {len(found_patients)} pacientes:")

                for pid, patient in found_patients:
                    with st.expander(f"{patient['name']} (ID: {pid})"):
                        self.show_patient_history(pid)
            else:
                st.warning("Nenhum paciente encontrado!")