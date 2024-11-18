import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys
import datetime
import json

# Adicionar o diretório src ao path
sys.path.append(str(Path(__file__).parent.parent.parent))


class HeartDiseasePredictor:
    def __init__(self):
        self.load_model()
        self.load_history()

    def load_model(self):
        """Carrega o modelo treinado"""
        try:
            model_path = Path(__file__).parent.parent.parent / 'models' / 'trained'
            latest_model = sorted(list(model_path.glob('random_forest*.joblib')))[-1]
            self.model = joblib.load(latest_model)
            st.sidebar.success(f"Modelo carregado: {latest_model.name}")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar modelo: {str(e)}")
            self.model = None

    def load_history(self):
        """Carrega histórico de predições"""
        history_path = Path(__file__).parent / 'prediction_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save_prediction(self, features, prediction):
        """Salva predição no histórico"""
        prediction_data = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': features,
            'prediction': float(prediction)
        }
        self.history.append(prediction_data)

        # Salvar no arquivo
        history_path = Path(__file__).parent / 'prediction_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

    def predict(self, features):
        """Faz predição usando o modelo carregado"""
        if self.model is None:
            return None

        # Criar DataFrame com os features
        df = pd.DataFrame([features])

        # Fazer predição
        prediction = self.model.predict_proba(df)[0]

        # Salvar predição
        self.save_prediction(features, prediction[1])

        return prediction[1]

    def plot_feature_importance(self, features):
        """Plota importância das features"""
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': list(features.keys()),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)

        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                     title='Importância das Features')
        return fig

    def plot_prediction_history(self):
        """Plota histórico de predições"""
        if not self.history:
            return None

        df = pd.DataFrame(self.history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = px.line(df, x='timestamp', y='prediction',
                      title='Histórico de Predições')
        return fig


def main():
    st.set_page_config(
        page_title="Previsão de Doenças Cardíacas",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inicializar predictor
    predictor = HeartDiseasePredictor()

    # Sidebar
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ["Fazer Predição", "Histórico", "Sobre"])

    if page == "Fazer Predição":
        st.title("❤️ Previsão de Doenças Cardíacas")
        st.markdown("""
        Este aplicativo utiliza Machine Learning para prever o risco de doenças cardíacas 
        com base em informações clínicas.
        """)

        # Criar três colunas para melhor organização
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Informações Pessoais")
            age = st.number_input("Idade", 20, 100, 50)
            sex = st.selectbox("Sexo", ["Masculino", "Feminino"])
            sex = 1 if sex == "Masculino" else 0

        with col2:
            st.subheader("Dados Clínicos")
            trestbps = st.number_input("Pressão Arterial (mm Hg)", 90, 200, 120)
            chol = st.number_input("Colesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Glicemia em Jejum > 120 mg/dl?", ["Não", "Sim"])
            fbs = 1 if fbs == "Sim" else 0

        with col3:
            st.subheader("Exames")
            restecg = st.selectbox(
                "ECG em Repouso",
                ["Normal", "Anormalidade ST-T", "Hipertrofia"]
            )
            restecg = {"Normal": 0, "Anormalidade ST-T": 1, "Hipertrofia": 2}[restecg]
            thalach = st.number_input("Freq. Cardíaca Máxima", 70, 220, 150)

        # Segunda linha de inputs
        col4, col5, col6 = st.columns(3)

        with col4:
            cp = st.selectbox(
                "Tipo de Dor no Peito",
                ["Típica Angina", "Atípica Angina", "Dor Não-Anginal", "Assintomático"]
            )
            cp = {
                "Típica Angina": 0,
                "Atípica Angina": 1,
                "Dor Não-Anginal": 2,
                "Assintomático": 3
            }[cp]

            exang = st.selectbox("Angina por Exercício?", ["Não", "Sim"])
            exang = 1 if exang == "Sim" else 0

        with col5:
            oldpeak = st.number_input("Depressão ST", 0.0, 6.0, 0.0)
            slope = st.selectbox(
                "Inclinação ST",
                ["Ascendente", "Plano", "Descendente"]
            )
            slope = {"Ascendente": 0, "Plano": 1, "Descendente": 2}[slope]

        with col6:
            ca = st.selectbox("Número de Vasos Principais", [0, 1, 2, 3])
            thal = st.selectbox(
                "Talassemia",
                ["Normal", "Defeito Fixo", "Defeito Reversível"]
            )
            thal = {"Normal": 1, "Defeito Fixo": 2, "Defeito Reversível": 3}[thal]

        # Botão de previsão
        if st.button("Realizar Previsão", type="primary"):
            features = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
                'ca': ca, 'thal': thal
            }

            prediction = predictor.predict(features)

            if prediction is not None:
                st.markdown("---")
                st.subheader("Resultado da Previsão")

                # Mostrar resultado em duas colunas
                col1, col2 = st.columns(2)

                with col1:
                    # Gauge chart para probabilidade
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilidade de Doença Cardíaca"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)

                with col2:
                    # Classificação de risco
                    risk_level = ("Alto" if prediction > 0.7 else
                                  "Médio" if prediction > 0.3 else "Baixo")
                    st.metric(
                        "Nível de Risco",
                        risk_level,
                        delta="Consulte um médico!" if risk_level == "Alto" else None
                    )

                    # Recomendações baseadas no risco
                    if risk_level == "Alto":
                        st.error("⚠️ Recomenda-se procurar atendimento médico o mais breve possível.")
                    elif risk_level == "Médio":
                        st.warning("⚠️ Considere agendar uma consulta médica para avaliação.")
                    else:
                        st.success("✅ Mantenha seus exames em dia e hábitos saudáveis.")

                # Plotar importância das features
                st.subheader("Fatores mais Influentes")
                fig = predictor.plot_feature_importance(features)
                if fig is not None:
                    st.plotly_chart(fig)

                st.info("""
                ℹ️ **Nota**: Esta é apenas uma previsão baseada em modelos estatísticos e não substitui 
                o diagnóstico médico profissional. Por favor, consulte um médico para uma avaliação adequada.
                """)

    elif page == "Histórico":
        st.title("📊 Histórico de Predições")
        fig = predictor.plot_prediction_history()
        if fig is not None:
            st.plotly_chart(fig)
        else:
            st.info("Nenhuma predição realizada ainda.")

    else:  # Página Sobre
        st.title("ℹ️ Sobre o Projeto")
        st.markdown("""
        ## Previsão de Doenças Cardíacas usando Machine Learning

        Este projeto utiliza técnicas avançadas de Machine Learning para prever o risco 
        de doenças cardíacas com base em diversos fatores clínicos.

        ### Modelo Utilizado
        - Random Forest Classifier
        - Acurácia: ~91%
        - Treinado com dados do UCI Heart Disease Dataset

        ### Features Utilizadas
        - Idade
        - Sexo
        - Tipo de dor no peito
        - Pressão arterial
        - Colesterol
        - E outros fatores clínicos importantes

        ### Limitações
        Este é um modelo preditivo e não deve ser usado como única fonte para 
        diagnóstico. Sempre consulte profissionais de saúde qualificados.

        ### Contato
        Para mais informações ou sugestões, entre em contato através do email: [seu-email]
        """)


if __name__ == "__main__":
    main()