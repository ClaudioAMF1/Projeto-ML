import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json


class RecommendationSystem:
    def __init__(self):
        self.load_recommendations()

    def load_recommendations(self):
        """Carrega base de recomendações"""
        self.recommendations = {
            'lifestyle': {
                'high_risk': [
                    "Procure atendimento médico imediatamente",
                    "Monitore sua pressão arterial diariamente",
                    "Evite atividades físicas intensas sem supervisão médica",
                    "Mantenha registro detalhado dos sintomas",
                    "Tome medicamentos conforme prescrição médica"
                ],
                'medium_risk': [
                    "Agende uma consulta médica nas próximas semanas",
                    "Comece um programa de exercícios leves",
                    "Reduza a ingestão de sal",
                    "Monitore sua pressão arterial semanalmente",
                    "Pratique técnicas de redução de estresse"
                ],
                'low_risk': [
                    "Mantenha hábitos saudáveis",
                    "Faça exercícios regulares",
                    "Mantenha uma dieta equilibrada",
                    "Faça checkups anuais",
                    "Monitore sua pressão arterial mensalmente"
                ]
            },
            'diet': {
                'high_risk': [
                    "Dieta com restrição severa de sódio (<1500mg/dia)",
                    "Evite totalmente gorduras trans",
                    "Limite gorduras saturadas a <7% das calorias",
                    "Priorize proteínas magras",
                    "Aumente consumo de fibras solúveis"
                ],
                'medium_risk': [
                    "Reduza sódio para <2300mg/dia",
                    "Limite gorduras saturadas",
                    "Aumente consumo de vegetais",
                    "Prefira grãos integrais",
                    "Modere consumo de açúcares"
                ],
                'low_risk': [
                    "Mantenha uma dieta balanceada",
                    "Consuma frutas e vegetais variados",
                    "Escolha grãos integrais",
                    "Modere consumo de álcool",
                    "Mantenha boa hidratação"
                ]
            },
            'exercise': {
                'high_risk': [
                    "Exercícios somente com supervisão médica",
                    "Foque em atividades de baixo impacto",
                    "Monitore frequência cardíaca",
                    "Evite exercícios em temperaturas extremas",
                    "Comece com 5-10 minutos por sessão"
                ],
                'medium_risk': [
                    "30 minutos de exercício moderado 5x/semana",
                    "Inclua caminhadas diárias",
                    "Faça exercícios de força leves",
                    "Pratique alongamentos",
                    "Monitore seu esforço"
                ],
                'low_risk': [
                    "150 minutos de exercício moderado por semana",
                    "Inclua exercícios aeróbicos",
                    "Faça treino de força 2-3x/semana",
                    "Pratique atividades variadas",
                    "Aumente gradualmente a intensidade"
                ]
            }
        }

    def get_risk_level(self, prediction: float) -> str:
        """Determina nível de risco baseado na predição"""
        if prediction > 0.7:
            return 'high_risk'
        elif prediction > 0.3:
            return 'medium_risk'
        return 'low_risk'

    def get_recommendations(self, prediction: float, features: dict) -> dict:
        """Gera recomendações personalizadas baseadas na predição e características"""
        risk_level = self.get_risk_level(prediction)

        recommendations = {
            'risk_level': risk_level,
            'lifestyle': self.recommendations['lifestyle'][risk_level],
            'diet': self.recommendations['diet'][risk_level],
            'exercise': self.recommendations['exercise'][risk_level]
        }

        # Adiciona recomendações específicas baseadas nas características
        specific_recommendations = []

        if features.get('age', 0) > 60:
            specific_recommendations.append(
                "Considerando sua idade, mantenha acompanhamento médico regular"
            )

        if features.get('trestbps', 0) > 140:
            specific_recommendations.append(
                "Sua pressão arterial está elevada. Monitore com frequência"
            )

        if features.get('chol', 0) > 200:
            specific_recommendations.append(
                "Seu colesterol está acima do ideal. Considere ajustes na dieta"
            )

        recommendations['specific'] = specific_recommendations
        return recommendations

    def show_recommendations(self, prediction: float, features: dict):
        """Mostra recomendações na interface"""
        recommendations = self.get_recommendations(prediction, features)

        # Título com cor baseada no risco
        risk_colors = {
            'high_risk': 'red',
            'medium_risk': 'orange',
            'low_risk': 'green'
        }

        st.markdown(
            f"<h2 style='color: {risk_colors[recommendations['risk_level']]}'>🎯 "
            f"Recomendações Personalizadas</h2>",
            unsafe_allow_html=True
        )

        # Tabs para diferentes categorias
        tab1, tab2, tab3 = st.tabs(["Estilo de Vida", "Dieta", "Exercícios"])

        with tab1:
            st.subheader("✨ Recomendações de Estilo de Vida")
            for rec in recommendations['lifestyle']:
                st.write(f"• {rec}")

        with tab2:
            st.subheader("🥗 Recomendações Nutricionais")
            for rec in recommendations['diet']:
                st.write(f"• {rec}")

        with tab3:
            st.subheader("🏃‍♂️ Recomendações de Exercícios")
            for rec in recommendations['exercise']:
                st.write(f"• {rec}")

        # Recomendações específicas
        if recommendations['specific']:
            st.subheader("📌 Recomendações Específicas")
            for rec in recommendations['specific']:
                st.info(rec)

        # Disclaimer
        st.markdown("""
        ---
        *Nota: Estas recomendações são baseadas em diretrizes gerais e seus fatores de risco específicos. 
        Sempre consulte um profissional de saúde antes de iniciar qualquer mudança significativa em sua rotina.*
        """)

    def export_recommendations(self, prediction: float, features: dict) -> str:
        """Exporta recomendações em formato de relatório"""
        recommendations = self.get_recommendations(prediction, features)

        report = f"""
        RELATÓRIO DE RECOMENDAÇÕES DE SAÚDE CARDIOVASCULAR
        ================================================
        Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}

        NÍVEL DE RISCO: {recommendations['risk_level'].upper().replace('_', ' ')}

        1. RECOMENDAÇÕES DE ESTILO DE VIDA
        ---------------------------------
        {chr(10).join('• ' + rec for rec in recommendations['lifestyle'])}

        2. RECOMENDAÇÕES NUTRICIONAIS
        ---------------------------
        {chr(10).join('• ' + rec for rec in recommendations['diet'])}

        3. RECOMENDAÇÕES DE EXERCÍCIOS
        ----------------------------
        {chr(10).join('• ' + rec for rec in recommendations['exercise'])}
        """

        if recommendations['specific']:
            report += f"""
        4. RECOMENDAÇÕES ESPECÍFICAS
        --------------------------
        {chr(10).join('• ' + rec for rec in recommendations['specific'])}
        """

        report += """
        OBSERVAÇÃO: Este relatório é gerado automaticamente com base em algoritmos 
        de avaliação de risco. Todas as recomendações devem ser discutidas e 
        validadas por um profissional de saúde qualificado.
        """

        return report