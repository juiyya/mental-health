# Classificação de Saúde Mental

Este repositório contém um estudo de **análise exploratória de dados (EDA)** e **pré-processamento** aplicado a um conjunto de dados de saúde mental, com foco em classificar o `Risco_saude_mental` em três níveis: **Baixo**, **Médio** e **Alto**.

O trabalho é feito em um notebook Jupyter, utilizando bibliotecas de ciência de dados em Python.

## Estrutura do projeto

- **CP1_2_3prompt.ipynb**: notebook principal com todo o fluxo de análise, pré-processamento e balanceamento dos dados.
- **README.md**: este arquivo, com visão geral do projeto.
- **requirements.txt**: lista de dependências Python usadas no notebook.

## Tecnologias utilizadas

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Como executar o notebook

1. **Clonar ou baixar** este repositório.
2. Garantir que o Python 3 está instalado.
3. (Opcional, mas recomendado) Criar um ambiente virtual:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   ```

4. Instalar as dependências:

   ```bash
   pip install -r requirements.txt
   ```

5. Abrir o Jupyter Notebook:

   ```bash
   jupyter notebook CP1_2_3prompt.ipynb
   ```

6. Ajustar o caminho do arquivo CSV, se necessário. No notebook, o dataset é carregado com:

   ```python
   df = pd.read_csv('caminho/para/mental_health_dataset.csv')
   ```

   Certifique-se de que o arquivo `mental_health_dataset.csv` esteja no caminho correto da sua máquina.

## Objetivos do projeto

- **Explorar** o conjunto de dados de saúde mental.
- **Tratar** e **limpar** dados (remoção de nulos e duplicados).
- **Balancear** as classes de risco de saúde mental.
- **Transformar** variáveis categóricas (binárias, ordinais e nominais) para uso em modelos de machine learning.

## Reprodutibilidade

Os passos do notebook foram organizados em seções numeradas (coleta de dados, análise exploratória, pré-processamento, etc.). Basta executar as células em ordem para reproduzir os resultados.

