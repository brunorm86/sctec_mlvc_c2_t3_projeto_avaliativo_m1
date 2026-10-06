# Machine Learning e Visão Computacional [T3] - Projeto Avaliativo
### Desenvolvimento de Pipeline Preditivo para Risco de Crédito

**Instituição:** SESI/SENAI
**Programa:** SCTEC
**Curso:** Machine Learning e Visão Computacional
**Turma:** 3 - Ciclo 2 (C2)
**Módulo:** 1 - Fundamentos de Programação, Dados e Machine Learning
**Aluno:** Bruno Ricardo Machado


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
5. Abra o arquivo `projeto_credito.ipynb` no VS Code ou inicie o Jupyter Lab.
6. Execute todas as células do notebook sequencialmente.

## 3. Estrutura de Versionamento e Branches (Git)
O desenvolvimento do projeto seguiu rigorosas práticas de controle de versão. Ao invés de um único commit massivo, adotamos um fluxo progressivo utilizando as seguintes ramificações (branches):
- `fase/eda`: Destinada à análise exploratória e visualizações.
- `fase/data-prep`: Destinada ao tratamento de nulos, duplicadas e remoção de outliers.
- `fase/modelagem`: Destinada ao balanceamento de classes (SMOTE), escalonamento e otimização de hiperparâmetros.

As ramificações foram iterativamente mescladas à branch `main`. Além disso, adotamos o padrão de **Commits Semânticos** (`feat:`, `fix:`, `docs:`) para garantir rastreabilidade e mensagens granulares no histórico.

## 4. Dicionário de Dados
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

## 5. Resumo Executivo e Veredito (Insights)

- **Análise Exploratória (EDA):** A investigação inicial revelou um forte desbalanceamento na variável alvo (78% de clientes adimplentes). Além disso, o uso de **Boxplots** detectou anomalias extremas e impossíveis nos dados, como solicitantes com 144 anos de idade e 123 anos de tempo de emprego formal.
- **Tratamento de Dados (Data Prep):** 
    - **Imputação Robusta:** Valores ausentes nas variáveis chave foram preenchidos com a **Mediana** em vez da Média. Essa decisão técnica protege a tendência central da amostra contra a distorção severa que seria causada pelos outliers.
    - **Impacto Algorítmico:** O tratamento minucioso de extremos foi imprescindível, especialmente porque algoritmos baseados em distância euclidiana, como o **KNN**, são altamente vulneráveis a variações de escala. Em contrapartida, a **Árvore de Decisão** apresentou-se inerentemente robusta a essas anomalias por utilizar particionamento monotônico.
- **Modelagem e Controle de Overfitting:** O monitoramento contínuo das métricas entre as bases de Treino e Teste norteou a otimização dos hiperparâmetros:
    - **KNN:** O modelo alcançou a melhor estabilidade em K=9, evitando a generalização enviesada de vizinhanças pequenas (Acurácia de Teste: 89.0%).
    - **Árvore de Decisão:** Árvores com profundidade ilimitada sofreram severo *overfitting* (alcançando 100% no Treino e perdendo eficácia no Teste). Restringir a profundidade com a poda estrutural de max_depth=7 ofereceu o melhor compromisso entre viés e variância (Acurácia de Teste: 90.6%).
- **Avaliação Avançada (ROC-AUC e Feature Importances):** Para reforçar a transparência exigida pelo setor de auditoria bancária, o gráfico de *Feature Importances* evidenciou que a renda comprometida e as taxas de juros são os fatores matemáticos determinantes para se assumir o risco de calote. Simultaneamente, a **Curva ROC-AUC** confirmou a superioridade discriminativa da Árvore de Decisão no balanço crítico entre Falsos Positivos e Falsos Negativos.
- **Veredito de Negócios:** Recomendamos fortemente a implantação da **Árvore de Decisão (max_depth=7)** em ambiente produtivo. Além de dispensar a etapa de escalonamento numérico e se mostrar resiliente aos outliers, o modelo assegura a **explicabilidade regulatória**. Acima de tudo, entregou um excelente **Recall (68%)** na identificação proativa de maus pagadores, mitigando ativamente a incidência do **Falso Negativo** — o erro operacional que resulta na perda irrecuperável do valor principal emprestado e que afeta diretamente a saúde financeira da instituição.

---

**Disclaimer:** Este projeto contou com o auxílio da inteligência artificial Google Gemini para suporte na sua codificação, estruturação e revisão, atuando como "pair programming" do aluno. Além disso, foi utilizada para gerar o conteúdo do arquivo Readme.md, organizar os commits (incluindo os padrões semânticos e de branch) e auxiliar em dúvidas pontuais e documentação do Python e bibliotecas usadas no projeto. 

---

### Dados do Autor
- **Nome:** Bruno Ricardo Machado
- **E-mail:** [brunorm869@gmail.com](mailto:brunorm869@gmail.com)
- **GitHub:** [@brunorm86](https://github.com/brunorm86)
