# CELL 0 (markdown)
# Projeto de Machine Learning: Risco de Crédito

"Este notebook implementa o pipeline preditivo para análise de risco de crédito, seguindo todas as fases exigidas."
----------------------------------------
# CELL 1 (markdown)
## Fase 1: Análise Exploratória de Dados (EDA)

Nesta primeira fase, vamos carregar a base de dados de Risco de Crédito e realizar uma análise descritiva e visual das variáveis.
----------------------------------------
# CELL 2 (code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de visualização
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Carregar os dados
df = pd.read_csv('./data/credit_risk_dataset.csv')

# Visualizar as primeiras linhas
display(df.head())
----------------------------------------
# CELL 3 (markdown)
### Análise Descritiva e Estatística
Vamos verificar o tamanho da base, os tipos de dados e o resumo estatístico.
----------------------------------------
# CELL 4 (code)
# Tamanho da base
print(f'Tamanho da base: {df.shape[0]} linhas e {df.shape[1]} colunas\n')

# Tipos de dados e nulos
print('Tipos de dados e valores não nulos:')
print(df.info())
print('\n----------------------------------\n')

# Sumário estatístico
print('Sumário Estatístico:')
display(df.describe())
----------------------------------------
# CELL 5 (markdown)
### Análise Visual
----------------------------------------
# CELL 6 (code)
# Gráfico 1: Desbalanceamento da Variável Alvo
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df, palette='Set2')
plt.title('Distribuição da Variável Alvo (loan_status)')
plt.xlabel('Status do Empréstimo (0 = Pago, 1 = Inadimplente)')
plt.ylabel('Contagem')
plt.show()

# Gráfico 2: Histograma de Idades
plt.figure(figsize=(8, 5))
sns.histplot(df['person_age'], bins=30, kde=True, color='skyblue')
plt.title('Distribuição da Idade dos Clientes')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# Gráfico 3: Mapa de Calor de Correlação de Pearson
plt.figure(figsize=(10, 8))
# Selecionar apenas colunas numéricas para a correlação
"num_df = df.select_dtypes(include=[np.number])\n",
"sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
"plt.title('Mapa de Calor de Correlação (Variáveis Numéricas)')\n",
"plt.show()\n",
"\n",
"# Gráfico 4: Identificação de Outliers via Boxplots (Idade e Tempo de Emprego)\n",
"fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
"sns.boxplot(x=df['person_age'], ax=axes[0], color='lightcoral')\n",
"axes[0].set_title('Outliers na Idade do Cliente')\n",
"axes[0].set_xlabel('Idade (Anos)')\n",
"\n",
"sns.boxplot(x=df['person_emp_length'], ax=axes[1], color='lightgreen')\n",
"axes[1].set_title('Outliers no Tempo de Emprego')\n",
"axes[1].set_xlabel('Tempo de Emprego (Anos)')\n",
"plt.tight_layout()\n",
"plt.show()"
----------------------------------------
# CELL 7 (markdown)
"### Tomada de Decisão (Parágrafo Analítico)\n",
"\n",
"Através da análise exploratória, constatamos que o dataset apresenta questões críticas a serem tratadas na fase de preparação:\n",
"\n",
"1. **Desbalanceamento Extremo:** A variável alvo (`loan_status`) é visivelmente assimétrica, com a maioria esmagadora dos clientes classificada como bons pagadores (0). Isso exige o uso de técnicas de balanceamento de classes como o **SMOTE** para evitar que o classificador aprenda um viés majoritário.\n",
"2. **Valores Faltantes (Nulos):** Há dados ausentes em `person_emp_length` e `loan_int_rate`. Precisamos imputar estes valores. Como as distribuições contêm outliers e assimetria, a **Mediana** é estatisticamente recomendada em detrimento da Média para reter a centralidade real.\n",
"3. **Presença de Outliers Críticos:** O gráfico 4 (Boxplots) revelou outliers gritantes de idade (144 anos) e de tempo de emprego (123 anos), que são inconsistentes com a vida humana e o histórico de crédito profissional. Esses outliers precisam ser removidos ou limitados porque distorcem de forma severa as distâncias euclidianas calculadas pelo modelo KNN.\n",
"4. **Correlações Relevantes:** O mapa de calor revela correlações expressivas entre a renda do cliente, o valor solicitado e a taxa de juros, o que guiará a criação de novas variáveis no Feature Engineering."
----------------------------------------
# CELL 8 (markdown)
## Fase 2: Tratamento e Limpeza (Data Prep)

Nesta fase, realizamos o tratamento e a limpeza da base de dados com as seguintes decisões fundamentadas estatisticamente:

1. **Remoção de Duplicadas:** Removemos registros duplicados que introduzem viés e redundância no aprendizado e avaliação dos modelos.
2. **Imputação de Valores Nulos (Mediana vs Média):** Optamos por preencher os valores nulos das variáveis person_emp_length e loan_int_rate com a **Mediana**. Justificativa: Ambas as colunas apresentam distribuições não normais e assimétricas, com presença de caudas longas (outliers). Imputar pela média causaria deslocamento artificial da tendência central por causa do peso excessivo dos outliers. A mediana é imune a esses extremos, mantendo o perfil estatístico natural da amostra.
3. **Remoção de Outliers Inconsistentes:** Excluímos idades superiores a 100 anos e tempo de emprego maior que 60 anos.

### *Nota Pedagógica: Impacto de Outliers nos Modelos*
- **KNN (K-Nearest Neighbors):** É extremamente sensível a outliers. Como se baseia em métricas de distância no espaço n-dimensional (como distância Euclidiana), um único valor absurdamente alto (ex. idade de 144) irá distorcer todas as escalas de distância. Isso causará classificações incorretas baseadas em vizinhos distantes.
- **Árvores de Decisão:** São inerentemente robustas a outliers. Elas realizam cortes binários simples e monotônicos (ex. renda superior a 50000). Esses cortes são baseados em ordenações e frequências de partição das classes, de forma que valores extremos nas pontas não afetam os limites definidos pelas regras mais internas da árvore.
----------------------------------------
# CELL 9 (code)
# 1. Remover duplicadas
print(f'Linhas antes de remover duplicadas: {df.shape[0]}')
df = df.drop_duplicates()
print(f'Linhas após remover duplicadas: {df.shape[0]}\n')

# 2. Tratar Nulos
# Identificar nulos
print('Valores nulos por coluna:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# Imputar nulos
# Usaremos a Mediana para person_emp_length e loan_int_rate, pois a mediana é menos sensível a outliers do que a média.
if 'person_emp_length' in df.columns:
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
if 'loan_int_rate' in df.columns:
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

print('\nNulos após imputação:')
print(df.isnull().sum().sum())

# 3. Tratar Outliers
# Remover idades extremas (ex: > 100 anos, pois são improváveis e distorcem o KNN)
df = df[df['person_age'] <= 100]
# Remover tempo de emprego irreal (ex: > 60 anos)
df = df[df['person_emp_length'] <= 60]

print(f'\nTamanho da base após tratamento de outliers: {df.shape[0]}')
----------------------------------------
# CELL 10 (markdown)
## Fase 3: Feature Engineering (Coluna Calculada)
Criaremos a coluna `comprometimento_renda = (loan_amnt / person_income) * 100`
----------------------------------------
# CELL 11 (code)
df['comprometimento_renda'] = (df['loan_amnt'] / df['person_income']) * 100
display(df[['loan_amnt', 'person_income', 'comprometimento_renda']].head())
----------------------------------------
# CELL 12 (markdown)
## Fase 4: Separação, Balanceamento e Escalonamento Seguro
Aqui realizaremos o encoding das variáveis categóricas, dividiremos os dados com `stratify=y`, balancearemos o conjunto de treino com `SMOTE` e escalonaremos as variáveis contínuas com `StandardScaler` apenas para o KNN.
----------------------------------------
# CELL 13 (code)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 1. Encoding (One-Hot Encoding para variáveis categóricas)
df_encoded = pd.get_dummies(df, drop_first=True)

# 2. Separação de X e y
X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

# Split com stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Balanceamento no Treino (SMOTE)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f'Proporção do alvo no treino antes do SMOTE:\n{y_train.value_counts()}')
print(f'\nProporção do alvo no treino após o SMOTE:\n{y_train_bal.value_counts()}')

# 4. Escalonamento (Apenas para o KNN, Árvore não precisa)
scaler = StandardScaler()
# Criamos versões de X escalonadas especificamente para o KNN
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Nota: A Árvore de Decisão utilizará X_train_bal e X_test (não escalonados), pois ela realiza cortes monotônicos baseados em regras (>, <) que não dependem de distância.
----------------------------------------
# CELL 14 (markdown)
## Fase 5: Modelagem e Validação (O Desafio do Overfitting)
Vamos treinar um modelo KNN (variando o K) e uma Árvore de Decisão (variando a profundidade) para diagnosticar e evitar o overfitting.
----------------------------------------
# CELL 15 (code)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Avaliando KNN (variando n_neighbors)
vizinhos = [3, 5, 7, 9]
knn_train_acc = []
knn_test_acc = []

for k in vizinhos:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_bal_scaled, y_train_bal)
    
    # Previsões
    y_pred_train_knn = knn.predict(X_train_bal_scaled)
    y_pred_test_knn = knn.predict(X_test_scaled)
    
    # Acurácia
    knn_train_acc.append(accuracy_score(y_train_bal, y_pred_train_knn))
    knn_test_acc.append(accuracy_score(y_test, y_pred_test_knn))

# Tabela de Resultados KNN
df_knn_results = pd.DataFrame({
    'K (Vizinhos)': vizinhos,
    'Acurácia Treino': knn_train_acc,
    'Acurácia Teste': knn_test_acc
})
print("================ TABELA DE RESULTADOS - KNN ================")
display(df_knn_results)
print("============================================================\\n")

# Plot KNN Overfitting
plt.figure(figsize=(8, 5))
plt.plot(vizinhos, knn_train_acc, label='Treino', marker='o')
plt.plot(vizinhos, knn_test_acc, label='Teste', marker='s')
plt.title('KNN - Diagnóstico de Overfitting (Curva de Complexidade)')
plt.xlabel('Número de Vizinhos (K)')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Melhor KNN escolhido visualmente para evitar overfitting (ex: K=9 que aproxima treino e teste)
best_knn = KNeighborsClassifier(n_neighbors=9)
best_knn.fit(X_train_bal_scaled, y_train_bal)
y_pred_best_knn = best_knn.predict(X_test_scaled)
----------------------------------------
# CELL 16 (code)
# 2. Avaliando Árvore de Decisão (variando max_depth)
profundidades = [3, 5, 7, None]
dt_train_acc = []
dt_test_acc = []

for depth in profundidades:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # Treino sem escalonamento!
    dt.fit(X_train_bal, y_train_bal)
    
    # Previsões
    y_pred_train_dt = dt.predict(X_train_bal)
    y_pred_test_dt = dt.predict(X_test)
    
    # Acurácia
    dt_train_acc.append(accuracy_score(y_train_bal, y_pred_train_dt))
    dt_test_acc.append(accuracy_score(y_test, y_pred_test_dt))

# Convertendo None para string para plotagem
profundidades_str = [str(d) for d in profundidades]

# Tabela de Resultados Árvore
df_dt_results = pd.DataFrame({
    'Profundidade (max_depth)': profundidades_str,
    'Acurácia Treino': dt_train_acc,
    'Acurácia Teste': dt_test_acc
})
print("================ TABELA DE RESULTADOS - ÁRVORE ================")
display(df_dt_results)
print("===============================================================\\n")

# Plot Decision Tree Overfitting
plt.figure(figsize=(8, 5))
plt.plot(profundidades_str, dt_train_acc, label='Treino', marker='o')
plt.plot(profundidades_str, dt_test_acc, label='Teste', marker='s')
plt.title('Árvore de Decisão - Diagnóstico de Overfitting')
plt.xlabel('Profundidade Máxima (max_depth)')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Melhor Árvore: None sofre extremo overfitting (treino vai para 1.0, teste cai).
# A profundidade 7 costuma ser o melhor equilíbrio.
best_dt = DecisionTreeClassifier(max_depth=7, random_state=42)
best_dt.fit(X_train_bal, y_train_bal)
y_pred_best_dt = best_dt.predict(X_test)
----------------------------------------
# CELL 17 (markdown)
## Fase 6: Avaliação e Veredito de Negócios
Agora que temos os dois melhores modelos treinados de forma balanceada e testados na base real, vamos analisar os erros (Matriz de Confusão).
----------------------------------------
# CELL 18 (code)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Matriz KNN
cm_knn = confusion_matrix(y_test, y_pred_best_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusão: Melhor KNN (K=9)')
axes[0].set_xlabel('Predito')
axes[0].set_ylabel('Real')

# Matriz Árvore
cm_dt = confusion_matrix(y_test, y_pred_best_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Matriz de Confusão: Melhor Árvore (Profundidade 7)')
axes[1].set_xlabel('Predito')
axes[1].set_ylabel('Real')

plt.tight_layout()
plt.show()

print('================ RELATÓRIO CLASSIFICAÇÃO: KNN =================')
print(classification_report(y_test, y_pred_best_knn))
print('\n\n================ RELATÓRIO CLASSIFICAÇÃO: ÁRVORE =================')
print(classification_report(y_test, y_pred_best_dt))
----------------------------------------
# CELL 19 (markdown)
### Veredito de Negócios

No setor financeiro (Risco de Crédito), precisamos entender o impacto de cada tipo de erro:
- **Falso Positivo (Previmos 1/Inadimplente, mas era 0/Bom Pagador)**: O banco deixa de emprestar para um cliente que pagaria. Custo: Perda da margem de lucro (juros que seriam recebidos).
- **Falso Negativo (Previmos 0/Bom Pagador, mas era 1/Inadimplente)**: O banco empresta para quem não vai pagar. Custo: Perda do **valor principal** emprestado (muito mais alto que os juros).

Logo, o erro mais grave para o banco é o **Falso Negativo**. O modelo ideal é aquele que consegue minimizar a proporção de Falsos Negativos (aumentando o Recall da classe 1), mesmo que isso gere mais Falsos Positivos.

Observando as matrizes de confusão e o Recall da classe 1:
Ambos os modelos tiveram um bom Recall graças ao balanceamento (SMOTE) no treino. A **Árvore de Decisão** tradicionalmente oferece não apenas boas métricas para dados não lineares sem precisar de escalonamento, mas também **interpretabilidade** (o banco pode explicar por que o crédito foi negado, algo exigido por reguladores). O KNN age como uma caixa-preta baseada em distâncias.

**Veredito:** Colocaria a **Árvore de Decisão (max_depth=7)** em produção, devido à sua facilidade de explicação para a área de negócios, robustez aos outliers identificados e excelente Recall para a classe de inadimplentes, minimizando os Falsos Negativos e, portanto, protegendo o capital do Banco.
----------------------------------------
