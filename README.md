# Projeto de Machine Learning: Risco de Crédito

Este repositório contém a Situação de Aprendizagem (Módulo 1) para a avaliação do Pipeline Preditivo em Ciência de Dados.

## 1. Descrição do Problema de Negócio
O banco/instituição financeira precisa de um modelo preditivo capaz de classificar clientes solicitantes de empréstimo entre:
- **0 (Bons Pagadores)**: O cliente pagará o empréstimo em dia.
- **1 (Inadimplentes / Calote)**: O cliente não pagará o empréstimo, gerando Risco de Crédito.

O objetivo de negócio é mitigar a inadimplência minimizando ao máximo os **Falsos Negativos** (prever que o cliente é bom pagador, quando na verdade ele dará calote), pois isso representa a perda do valor principal emprestado, causando grande prejuízo financeiro.

## 2. Instruções de Instalação e Uso
O projeto foi desenvolvido inteiramente em Python no formato de Jupyter Notebook para facilitar a leitura das justificativas teóricas e a renderização dos gráficos de Análise Exploratória (EDA).

1. Clone o repositório ou baixe os arquivos.
2. Certifique-se de ter o Python 3.9+ instalado.
3. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   ```
4. Ative o ambiente virtual e instale as dependências:
   ```bash
   # No Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   
   pip install -r requirements.txt
   ```
5. Baixe a base de dados utilizando o módulo gdown (já incluso no requirements):
   ```bash
   gdown 1R4WCMc_56lv3fMaDalUBAz5jSxeZOcwI -O credit_risk_dataset.csv
   ```
6. Abra o arquivo `projeto_credito.ipynb` no VS Code ou inicie o Jupyter Lab.
7. Execute todas as células do notebook sequencialmente.

## 3. Dicionário de Dados
Abaixo estão as principais variáveis e a nova variável calculada criada na Fase 3 do projeto:

- `person_age`: Idade da pessoa.
- `person_income`: Renda anual da pessoa.
- `person_home_ownership`: Status de moradia (Aluguel, Própria, Hipoteca).
- `person_emp_length`: Tempo de emprego em anos.
- `loan_intent`: Motivo do empréstimo (Educação, Médico, Empreendimento).
- `loan_grade`: Grau de risco atribuído historicamente pela instituição.
- `loan_amnt`: Valor total do empréstimo solicitado.
- `loan_int_rate`: Taxa de Juros do empréstimo.
- `loan_percent_income`: Percentual da renda comprometido (variável original).
- `cb_person_default_on_file`: Possui histórico de inadimplência prévio (Y/N).
- `cb_person_cred_hist_length`: Tempo de histórico de crédito.
- `loan_status` (Variável Alvo): 0 se pagou, 1 se ficou inadimplente.

**Variável Gerada no Feature Engineering:**
- `comprometimento_renda`: Calculado pela fórmula `(loan_amnt / person_income) * 100`. Indica o real impacto mensal/anual da dívida sobre a renda total, refinando a variável percentual existente.

## 4. Resumo Executivo e Veredito (Insights)

- **Análise Exploratória (EDA) & Visualização de Outliers:** Identificamos forte desbalanceamento na variável alvo, com cerca de 78% dos dados concentrados em bons pagadores. Adicionamos a plotagem de **Boxplots** (Gráfico 4) que comprovou a presença de outliers extremos e impossíveis na idade (144 anos) e tempo de emprego (123 anos).
- **Decisão de Tratamento de Dados (Data Prep):** 
    - **Imputação Robustecida:** Os nulos de person_emp_length e loan_int_rate foram tratados com a **Mediana** (em detrimento da Média), uma vez que a distribuição apresenta assimetria e caudas longas. A mediana protege a tendência central sem distorcê-la por pesos extremos.
    - **Impacto nos Modelos:** Documentamos a justificativa pedagógica de que o **KNN** é altamente vulnerável a outliers devido ao cálculo de distâncias euclidianas n-dimensionais (que mudam de escala severamente), enquanto a **Árvore de Decisão** é imune a extremos porque utiliza cortes monotônicos binários ordenados de classes.
- **Modelagem e Otimização com Tabelas e Gráficos:** Para o combate ao overfitting, monitoramos de perto a complexidade em **gráficos de curva e tabelas de resultados de Treino vs. Teste simultâneos**:
    - **KNN:** Estabilizou no parâmetro ideal de K=9 (Acurácia de Treino: 92.8% | Teste: 89.0%), evitando a decoreba de vizinhos muito próximos.
    - **Árvore de Decisão:** Sofre overfitting absoluto (Acurácia de treino 100.0% e queda de teste) se a profundidade for livre (None). A restrição robusta para max_depth=7 garantiu o melhor ponto de equilíbrio e generalização (Acurácia de Treino: 90.4% | Teste: 90.6%).
- **Veredito de Negócios:** Recomendamos a **Árvore de Decisão (max_depth=7)** para implantação em produção. Ela oferece a explicabilidade regulatória exigida pelas auditorias de risco bancário (regras condicionais claras), dispensa escalonamento numérico e manteve um excelente **Recall (68%)** de classe 1 (inadimplentes), mitigando eficientemente o erro financeiro mais letal para a instituição: o **Falso Negativo** (emprestar valor principal a um mau pagador).