# Dicionário de Dados - Risco de Crédito

Este documento explica as variáveis presentes no dataset de risco de crédito (`credit_risk_dataset.csv`).

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| **person_age** | Numérica (int) | Idade do cliente (em anos). |
| **person_income** | Numérica (int) | Renda anual do cliente. |
| **person_home_ownership** | Categórica (str) | Situação de moradia do cliente (`RENT` = Aluguel, `OWN` = Própria, `MORTGAGE` = Financiada, `OTHER` = Outro). |
| **person_emp_length** | Numérica (float) | Tempo de emprego atual do cliente (em anos). |
| **loan_intent** | Categórica (str) | Propósito do empréstimo (`PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`). |
| **loan_grade** | Categórica (str) | Classificação/nota de risco do empréstimo atribuída pela instituição (categorias de `A` a `G`). |
| **loan_amnt** | Numérica (int) | Valor total do empréstimo solicitado. |
| **loan_int_rate** | Numérica (float) | Taxa de juros aplicada ao empréstimo. |
| **loan_status** | Numérica (int) | Status do empréstimo (**Variável Alvo**). `0` = Empréstimo não teve calote (pago) e `1` = Inadimplência (calote). |
| **loan_percent_income** | Numérica (float) | Razão entre o valor do empréstimo e a renda anual do cliente. |
| **cb_person_default_on_file** | Categórica (str) | Indica se o cliente possui histórico prévio de inadimplência registrado (`Y` = Sim, `N` = Não). |
| **cb_person_cred_hist_length** | Numérica (int) | Tempo de histórico de crédito do cliente (em anos). |
