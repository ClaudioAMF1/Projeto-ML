import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
import base64
import json
import yaml
from datetime import datetime
import streamlit_authenticator as stauth


def load_auth_config():
    """Carrega configuração de autenticação"""
    config_path = Path(__file__).parent / 'auth_config.yaml'
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def create_authenticator():
    """Cria autenticador do Streamlit"""
    config = load_auth_config()
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    return authenticator


def export_to_excel(df, filename):
    """Exporta DataFrame para Excel"""
    output = Path(__file__).parent.parent / 'assets' / filename
    df.to_excel(output, index=False)
    return output


def create_download_link(file_path, link_text):
    """Cria link de download para arquivo"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'data:application/octet-stream;base64,{b64}'
    return f'<a href="{href}" download="{file_path.name}">{link_text}</a>'


def plot_risk_gauge(value, title="Risco de Doença Cardíaca"):
    """Cria gráfico gauge para visualização de risco"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="lavender",
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def create_comparison_chart(patient_data, population_data):
    """Cria gráfico de comparação entre paciente e população"""
    df_comparison = pd.DataFrame({
        'Paciente': patient_data,
        'Média População': population_data
    }).transpose()

    fig = px.bar(df_comparison, barmode='group',
                 title='Comparação com a População')
    return fig


def generate_report(patient_data, prediction, feature_importance):
    """Gera relatório em HTML"""
    report = f"""
    <h1>Relatório de Avaliação de Risco Cardíaco</h1>
    <p>Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>

    <h2>Dados do Paciente</h2>
    <table>
        {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in patient_data.items())}
    </table>

    <h2>Resultado da Avaliação</h2>
    <p>Probabilidade de Doença Cardíaca: {prediction:.1%}</p>
    <p>Nível de Risco: {'Alto' if prediction > 0.7 else 'Médio' if prediction > 0.3 else 'Baixo'}</p>

    <h2>Fatores mais Importantes</h2>
    <table>
        {''.join(f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in feature_importance.items())}
    </table>

    <p><em>Este relatório é gerado automaticamente e não substitui avaliação médica profissional.</em></p>
    """
    return report


class SessionState:
    """Classe para gerenciar estado da sessão"""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def get_session_state(**kwargs):
    """Obtém ou cria estado da sessão"""
    if not hasattr(st, '_session_state'):
        st._session_state = SessionState(**kwargs)
    return st._session_state


def load_css():
    """Carrega CSS customizado"""
    css = """
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
        }
        .reporting-widget {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .custom-metric {
            padding: 1rem;
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)