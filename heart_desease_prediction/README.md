# Sistema de Previsão de Doenças Cardíacas 🫀

## Descrição
Este projeto implementa um sistema de Machine Learning para prever o risco de doenças cardíacas com base em informações clínicas. O sistema inclui uma interface web interativa, visualizações avançadas e recomendações personalizadas.

## Funcionalidades Principais 🌟
- Previsão de risco cardíaco em tempo real
- Visualização interativa dos resultados
- Análise dos fatores mais influentes
- Recomendações personalizadas baseadas no nível de risco
- Interface web intuitiva e responsiva

## Tecnologias Utilizadas 🛠
- **Python 3.9+**
- **Frameworks e Bibliotecas**:
  - Streamlit (Interface Web)
  - Scikit-learn (Machine Learning)
  - Pandas (Manipulação de Dados)
  - Plotly (Visualizações)
  - NumPy (Computação Numérica)
  - FPDF (Geração de PDFs)

## Estrutura do Projeto 📁
```
heart_disease_prediction/
├── data/                    # Dados brutos e processados
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/                  # Modelos treinados
│   ├── trained/
│   └── checkpoints/
├── src/                    # Código fonte
│   ├── data/               # Scripts de processamento de dados
│   ├── models/             # Scripts de treinamento
│   ├── visualization/      # Scripts de visualização
│   └── web/               # Interface web
├── tests/                  # Testes unitários
├── configs/                # Arquivos de configuração
└── reports/               # Relatórios gerados
```

## Como Instalar e Executar 🚀

### Pré-requisitos
- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)

### Instalação
1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd heart_disease_prediction
```

2. Crie um ambiente virtual:
```bash
python -m venv .venv
```

3. Ative o ambiente virtual:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Executando o Projeto
1. Treine o modelo:
```bash
python src/models/train.py
```

2. Inicie a interface web:
```bash
streamlit run src/web/app.py
```

## Como Usar 📱
1. Acesse a interface web através do navegador
2. Preencha os dados clínicos solicitados
3. Clique em "Realizar Previsão"
4. Analise os resultados e recomendações
5. Exporte relatórios se necessário

## Features do Modelo 📊
- **Entradas**:
  - Idade
  - Sexo
  - Tipo de dor no peito
  - Pressão arterial
  - Colesterol
  - Glicemia em jejum
  - ECG em repouso
  - Frequência cardíaca máxima
  - Outros fatores clínicos

- **Saídas**:
  - Probabilidade de doença cardíaca
  - Nível de risco (Baixo, Médio, Alto)
  - Fatores mais influentes
  - Recomendações personalizadas

## Performance do Modelo 📈
- Acurácia: ~91%
- Precisão: 92.27%
- Recall: 91.80%
- F1-Score: 91.82%

## Próximos Passos 🎯
- [ ] Implementar autenticação de usuários
- [ ] Adicionar mais visualizações
- [ ] Integrar com outros dados clínicos
- [ ] Melhorar as recomendações
- [ ] Adicionar suporte a múltiplos idiomas

## Autores 🙋🏼‍♂️ 
- Claudio Meireles
- Kelwin Menezes
