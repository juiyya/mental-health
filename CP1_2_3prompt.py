import pandas as pd #manipulacao de data frame
import numpy as np #manipulacao de arrays
import matplotlib.pyplot as plt # graficos
import seaborn as sns #graficos
import time
from sklearn.utils import resample # balanceamento de dados
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 3.3.3 Atributos Nominais e 3.5 Padronização Z-Score para melhorar o uso com a ia transformando em 0 e 1
from sklearn.model_selection import GridSearchCV, train_test_split  # GridSearchCV: Otimização de hiperparâmetros | train_test_split: Divisão de dados em treino e teste | Usado no 5.0
from sklearn.tree import DecisionTreeClassifier, plot_tree # arvore de decisao
from sklearn.metrics import confusion_matrix, classification_report, recall_score # matriz e metricas de validação

# carrega o arquibo csv do dataset para a criação
df = pd.read_csv('C:\FIAP\CP123_prompt\mental_health.dataset.csv') # diretório do arquivo
df.head()

"""###1.1 mudar nome das colunas para portugues

"""

df.rename(columns={'age': 'Idade',
                   'gender': 'Genero',
                   'employment_status' : 'Status_empregado',
                   'work_environment' : 'Ambiente_de_trabalho',
                   'mental_health_history' : 'Historico_saude_mental',
                   'seeks_treatment' : 'Busca_tratamento',
                   'stress_level' : 'Nivel_stress',
                   'sleep_hours' : 'Horas_sono',
                   'physical_activity_days' : 'Dias_exercicio',
                   'depression_score' : 'Depressao',
                   'anxiety_score' : 'Ansiedade',
                   'social_support_score' : 'Suporte_social',
                   'productivity_score' : 'Produtividade',
                   'mental_health_risk' : 'Risco_saude_mental'

                   }, inplace=True)
df.head()

"""###1.2 mudar valores únicos

"""

# valores unicos
df['Genero'].unique()

#renomear uma ou mais iformaçõesdo dataframe
# a troca é realizada entre as chaves e os valores do dicinário
substituicoes = {
    'Male' : 'M',
    'Female' : 'F',
    'Non-binary' : 'O',
    'Prefer not to say' : 'O'
}
df = df.replace(substituicoes)
df.head()

#df["Status_empregado"].value_counts()
df["Status_empregado"].unique()

substituicoes0 = {
    'Employed' : 'Empregado',
    'Student' : 'Estudante',
    'Self-employed' : 'Autonomo',
    'Unemployed' : 'Desempregado'
}
df = df.replace(substituicoes0)
df.head()

df["Ambiente_de_trabalho"].unique()

substituicoes1 = {
    'On-site' : 'Presencial',
    'Remote' : 'Remoto',
    'Hybrid' : 'Hibrido'
}
df = df.replace(substituicoes1)
df.head()

substituicoes2 = {
    'Yes' : 'Sim',
    'No' : 'Nao'
}
df = df.replace(substituicoes2)
df.head()

df['Risco_saude_mental'].unique()

substituicoes3 = {
    'High' : 'Alto',
    'Medium' : 'Medio',
    'Low' : 'Baixo'
}
df = df.replace(substituicoes3)
df.head()

"""###1.3 lista"""

# Categoricos (qualificativos ou classificativos)
atributos_categoricos = ['Genero', 'Status_empregado', 'Ambiente_de_trabalho',
                          'Busca_tratamento', 'Risco_saude_mental']
# Numericos (quantitativos)
atributos_numericos = ['Idade', 'Nivel_stress', 'Horas_sono', 'Dias_exercicio', 'Depressao', 'Ansiedade', 'Suporte_social', 'Produtividade']

# Binarios
atributos_categoricos_binarios = ['Historico_saude_mental', 'Busca_tratamento']

# Nominais
atributos_categoricos_nominais = ['Genero', 'Status_empregado', 'Ambiente_de_trabalho']

# Ordinal
atributos_categoricos_ordinais = ['Risco_saude_mental']

# Rotulo
rotulo = 'Risco_saude_mental'

"""##2. análise exploratória dos dados

### 2.1 indicadores
"""

#indicadores das variaveis numericas para pessoas saudaveis
df[atributos_numericos][df['Risco_saude_mental'] == 'Alto'].describe()

#indicadores das variaves numericas
df[atributos_numericos][df['Risco_saude_mental'] == 'Medio'].describe()

# indicadores das variaveis numericas
df[atributos_numericos][df['Risco_saude_mental'] == 'Baixo'].describe()

# por padrao describe so filtra numericos por isso incluimos ele como object
# categoricos so lida com string, se nao for string usar o df.types para verificar qual adulterado e modificar para umas string
df[atributos_categoricos].describe(include=['object'])

# indicadores das variáveis categóricas para pessoas saudáveis
df[atributos_categoricos][df['Genero'] == 'M'].describe(include=['object'])

# indicadores das variáveis categóricas para pessoas saudáveis
df[atributos_categoricos][df['Genero'] == 'F'].describe(include=['object'])

# indicadores das variáveis categóricas para pessoas saudáveis
df[atributos_categoricos][df['Risco_saude_mental'] == 'Baixo'].describe(include=['object'])

# indicadores das variáveis categóricas para pessoas meio saudáveis
df[atributos_categoricos][df['Risco_saude_mental'] == 'Medio'].describe(include=['object'])

# indicadores das variáveis categóricas para pessoas não saudáveis
df[atributos_categoricos][df['Risco_saude_mental'] == 'Alto'].describe(include=['object'])

print('Contagem das classes dos atributos categoricos das pessoas\n')
for atributos_categorico in atributos_categoricos:
    print('Risco de doença mental baixo')
    print(df[atributos_categorico][df['Risco_saude_mental'] == 'Baixo'].value_counts(),'\n')
    print('Risco de doença mental moderado')
    print(df[atributos_categorico][df['Risco_saude_mental'] == 'Medio'].value_counts(),'\n')
    print('Risco de doença mental alto')
    print(df[atributos_categorico][df['Risco_saude_mental'] == 'Alto'].value_counts(),'\n\n\n')

"""### 2.2 gráficos"""

#Gráfico Contigência dos atributos categóricos
#for atributo_categorico in atributos_categoricos:
    #pd.crosstab(df[atributo_categorico], df[rotulo]).plot(kind='bar', stacked=True, figsize=(5, 5))
    #plt.title(atributo_categorico, fontweight='bold')
    #plt.legend(title=rotulo[0], bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    #plt.show()

#Gráfico Contigência dos atributos categóricos
num_atributos_cat = len(atributos_categoricos)
ncols_cat = 3 # Número de colunas para os subplots (ajuste conforme necessário)
nrows_cat = (num_atributos_cat + ncols_cat - 1) // ncols_cat # Calcula o número de linhas necessário

fig_cat, axes_cat = plt.subplots(nrows=nrows_cat, ncols=ncols_cat, figsize=(18, nrows_cat * 5)) # Ajuste o tamanho da figura
axes_cat = axes_cat.flatten() # Transforma a matriz de eixos em um array 1D

for i, atributo_categorico in enumerate(atributos_categoricos):
    pd.crosstab(df[atributo_categorico], df[rotulo]).plot(kind = 'bar', stacked = True, ax=axes_cat[i])
    axes_cat[i].set_title(atributo_categorico, fontweight='bold')
    axes_cat[i].set_xlabel(atributo_categorico)
    axes_cat[i].set_ylabel('Contagem')
    axes_cat[i].legend(title = rotulo, bbox_to_anchor=(1.05, 1), loc = 'upper left')


# Remove subplots vazios, se houver
for j in range(i + 1, len(axes_cat)):
    fig_cat.delaxes(axes_cat[j])


plt.tight_layout()
plt.show()

# Gráficos Boxplots dos atributos numéricos para pessoas saudáveis
#for atributo_numerico in atributos_numericos:
  #plt.figure(figsize=(4, 3))
  #sns.boxplot(x=rotulo, y=atributo_numerico, data=df[[rotulo, atributo_numerico]])
  #plt.title('Distribuição de ' + atributo_numerico + ' de pessoas saudáveis')
  #plt.show()

# Gráficos Boxplots dos atributos numéricos
num_atributos = len(atributos_numericos)
ncols = 4 # Número de colunas para os subplots (ajuste conforme necessário)
nrows = (num_atributos + ncols - 1) // ncols # Calcula o número de linhas necessário

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, nrows * 4)) # Ajuste o tamanho da figura conforme necessário
axes = axes.flatten() # Transforma a matriz de eixos em um array 1D

for i, atributo_numerico in enumerate(atributos_numericos):
    sns.boxplot(x=rotulo, y=atributo_numerico, data=df[[rotulo, atributo_numerico]], ax=axes[i])
    axes[i].set_title('Distribuição de ' + atributo_numerico)
    axes[i].set_xlabel(rotulo)
    axes[i].set_ylabel(atributo_numerico)

# Remove subplots vazios, se houver
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Correlação entre os atributos numéricos
plt.figure(figsize=(7, 7)) # tamanho do gráfico, largura X altura
matriz_corr = df[atributos_numericos].corr() # trabalha apenas com numeros, nao trabalha com categoricos, .corr() já cria a correlção
sns.heatmap(matriz_corr,annot=True,cmap='coolwarm',fmt='.2f') # mapa de calor, numero dentro das células = "annot = True" / cmap = mudar a cor / fmt = casas apos virgula
plt.title('Correlação entre Atributos númericos', fontweight='bold') # titulo do grafico
plt.show() # mostra o grafico

"""## 3. pré processamentos

### 3.1 limpeza de dados
"""

# Remove amostras com dados faltantes
df_sem_NAN  = df.dropna(axis=0) #remove linhas com dados faltantes
print('Quantidade de linhas com dados faltantes por atributo:')
df_sem_NAN.isnull().sum()      #soma de dados faltantes por atributo

# Remove amostras duplicadas
df_sem_duplicata  = df_sem_NAN .drop_duplicates()
qtd_duplicatas    = str(df_sem_duplicata.duplicated().sum())
print('Quantidade de amostras duplicadas: ' + qtd_duplicatas)

df_limpo = df_sem_duplicata.copy()

"""### 3.2 balanceamento de dados"""

# verifica a quantidade de amostras por classe do rótulo
df_limpo[rotulo].value_counts()

# Realiza o balanceamento do dataset através do undersampling
# algumas amostras da classe majoritária são retiradas
df_major = df_limpo[df_limpo[rotulo] == 'Medio']  # dataframe com amostras da classe marjoritária
df_minor = df_limpo[df_limpo[rotulo] == 'Baixo']  # dataframe com amostras da classe minoritária

df_major_reduzido = resample(df_major,                # dataframe com as amostras da classe majoritária
                             replace=False,           # False (undersampling), True (Oversampling)
                             n_samples=len(df_minor), # quantidade de amostras da classe minoritária
                             random_state=42)         # pseudoaleatório

# cria dataframe balanceado
df_balanceado = pd.concat(                    # pd.concat une os dataframes
                          [df_major_reduzido, # dataframe com amostras da classe marjoritária
                           df_minor]          # dataframe com amostras da classe minoritária
                         )

print("Distribuição das categorias balanceadas:\n")
print(df_balanceado[rotulo].value_counts())

# Realiza o balanceamento do dataset através do undersampling
# algumas amostras da classe majoritária são retiradas
df_major = df_limpo[df_limpo[rotulo] == "Medio"]  # Dataframe com amostras da classe marjoritária
df_medio = df_limpo[df_limpo[rotulo] == "Alto"]   # Dataframe com amostras da classe média
df_minor = df_limpo[df_limpo[rotulo] == "Baixo"]  # Dataframe com amostras da classe minoritária


df_major_reduzido = resample(df_major,                 # dataframe com as amostras da classe majoritária
                             replace=False,            # False (undersampling - retira até ficar com a menor), True (Oversampling - cria dados artificiais, bom para cadastro facial por exemplo)
                             n_samples=len(df_minor),  # quantidade de amostras da classe minoritária
                             random_state=42)          # pseudoaleatório - Número aleatório mas na mesma ordem para quem tiver acesso a este notebook, mesmo resultado a todas as pessoas

df_medio_reduzido = resample(df_medio,                 # dataframe com as amostras da classe média
                             replace=False,            # False (undersampling - retira até ficar com a menor), True (Oversampling - cria dados artificiais, bom para cadastro facial por exemplo)
                             n_samples=len(df_minor),  # quantidade de amostras da classe minoritária
                             random_state=42)          # pseudoaleatório - Número aleatório mas na mesma ordem para quem tiver acesso a este notebook, mesmo resultado a todas as pessoas


# cria dataframe balanceado
df_balanceado = pd.concat(                             # pd.concat une os dataframes
                          [df_major_reduzido,          # dataframe com amostras da classe marjoritária
                           df_medio_reduzido,          # dataframe com amostras da classe média
                           df_minor]                   # dataframe com amostras da classe minoritária
                         )

print("Distribuição das categorias balanceadas:\n")
print(df_balanceado[rotulo].value_counts())

"""### 3.3 transformação atributos catgóricos

#### 3.3.1 binarios
"""

print(df_balanceado['Historico_saude_mental'].value_counts())
print(df_balanceado['Busca_tratamento'].value_counts())

# Transformação de variaveis categóricas binárias
substituicoes = {
    'Nao': 0,
    'Sim': 1
}

df_binarias_transformadas = df_balanceado[atributos_categoricos_binarios].replace(substituicoes)
df_binarias_transformadas.head()

"""#### 3.3.2 ordinais"""

print(df_balanceado[atributos_categoricos_ordinais].value_counts())

# Transformação de variaveis categóricas ordinais
substituicoes = {
    'Baixo': 0,
    'Medio': 1,
    'Alto': 2
}

df_ordinais_transformadas = df_balanceado[atributos_categoricos_ordinais].replace(substituicoes)
df_ordinais_transformadas.head()

"""#### 3.3.3 nominais"""

print(df_balanceado[atributos_categoricos_nominais].head())

# transformação de variaveis categóricas nominais através da classe OneHotEncoder
# novas colunas a partir dos grupos existentes nos atributos categóricos
# será uma coluna nova para cada grupo
# isso garante que atributos categóricos nominais não tenham relação de grandeza

encoder = OneHotEncoder(sparse_output=False) # garante que o dado seja numpy array
encoded_array = encoder.fit_transform(df_balanceado[atributos_categoricos_nominais]) # transforma dados

novas_colunas = encoder.get_feature_names_out(atributos_categoricos_nominais) # nome das novas colunas

df_nominais_transformadas = pd.DataFrame(
                                        encoded_array,
                                        columns= novas_colunas,
                                        index=df_balanceado.index # Use the index from the balanced DataFrame
                                        )
df_nominais_transformadas.head()

"""### 3.4 transformação de rótulo"""

print(df_balanceado[rotulo].unique())

substituicoes = {
    'Alto': 0,
    'Medio': 1,
    'Baixo': 2
}
df_rotulo_transformado = df_balanceado[rotulo].replace(substituicoes)

df_rotulo_transformado.head()

# Une todas transformações realizadas em um único dataframe
df_transformado = pd.concat(                                        # pd.concat realiza a união dos dataframes
                              [df_balanceado[atributos_numericos],  # dataframe com atributos numéricos
                               df_binarias_transformadas,           # dataframe com atributos categóricos binarias
                               df_nominais_transformadas,           # dataframe com atributos categóricos nominais
                               df_ordinais_transformadas,           # dataframe com atributos categóricos ordinais
                               df_rotulo_transformado               # rótulo df_limpo[rotulo] para Regressão
                              ],
                              axis=1                                # acessa as colunas dos dataframes
                            )

df_transformado.head()

"""### 3.5 padronização Z-score
coloca os atributos numericos na mesma escala (media=0, desvio padrao=1)
"""

# converte as colunas para float, auxilia na conversao z-score e no treinamento
df_float = df_transformado.astype(float)

# dataframe sem padronizacao z-score
df_preprocessado_nao_padronizado = df_float.copy()

# z-score das variaveis numericas atraves do StandardScaler()
# garante que as numericas estajam na mesma escala
scaler = StandardScaler()

df_preprocessado_padronizado = df_preprocessado_nao_padronizado.copy() # copia de um p outro
z = scaler.fit_transform(df_preprocessado_nao_padronizado[atributos_numericos]) # z-score numericos
df_preprocessado_padronizado.loc[:, atributos_numericos] = z

df_preprocessado_padronizado.head()

#indicadores dos atributos numéricos após o Z-score
df_preprocessado_padronizado[atributos_numericos].describe()

"""## 4. divisão de dados
divisao dos dados em treino e teste para os treinamentos
"""

atributos = df_preprocessado_nao_padronizado.drop(columns=[rotulo]).columns.tolist()

"""### 4.1 não padronizados"""

# Divide os dados não padronizados em treino(70%) e teste(30%)
# alguns modelos (Árvore de decisão, Random Forest e outros) utilizam dados NÃO padronizados no treino
divisao_nao_padronizado = train_test_split(df_preprocessado_nao_padronizado[atributos],
                                            df_preprocessado_nao_padronizado[rotulo],
                                            test_size=0.3,            # proporção treino teste. Ex: 0.3 = 30% de teste
                                            random_state=42,          # pseudoaleatório
                                            stratify=df_preprocessado_nao_padronizado[rotulo])      # amostras de teste são balanceadas a partir das classes dos rótulos

atributos_treino_nao_pad, atributos_teste_nao_pad, rotulo_treino_nao_pad, rotulo_teste_nao_pad = divisao_nao_padronizado  # armazena os grupos de amostras (treino e teste)

"""### 4.2 padronizados"""

# Divide os dados padronizados em treino(70%) e teste(30%)
# alguns modelos (SVM, k-NN, Redes Neurais e outros) utilizam dados padronizados no treino
divisao_padronizado = train_test_split(df_preprocessado_padronizado[atributos],       # atributos
                                       df_preprocessado_padronizado[rotulo],          # rótulo
                                       test_size=0.3,                                 # proporção treino teste. Ex: 0.3 = 30% de teste
                                       random_state=42,                               # pseudoaleatório
                                       stratify=df_preprocessado_padronizado[rotulo]) # amostras de teste são balanceadas a partir das classes dos rótulos

atributos_treino_pad, atributos_teste_pad, rotulo_treino_pad, rotulo_teste_pad = divisao_padronizado  # armazena os grupos de amostras (treino e teste)

"""## 5. construção dos modelos e treino

### 5.1 decision tree
"""

decision_tree_classifier = DecisionTreeClassifier(random_state = 42)  # import

# Hiperparâmetros do GridSearchCV
# GridSearchCV testa diversos hiperparâmetros no treinamento. Após isso, encontra os melhores parâmetros

param_grid_ad = {
    'criterion'         : ['gini', 'entropy'],      # critérios de divisão de dados
    'max_depth'         : [10, 12, 14],             # quantidade de níveis (desde o nó raiz até a última folha)
    'min_samples_split' : [6, 8],                   # quantidade mínima de amostras por nó
    'min_samples_leaf'  : [4, 6, 8, 10],            # quantidade mínima de amostras de um tipo por nó
    'max_features'      : [None, 'sqrt', 'log2']    # método para definir atributos de cada treino
}

# Instância do GridSearchCV (Árvore de decisão)
grid_search_ad=GridSearchCV(
    estimator=decision_tree_classifier,  # instância do modelo
    param_grid=param_grid_ad,  # parâmetros GridSearchCV
    cv=5,  # cross validation
    scoring='recall',  # metricas de validacao
    n_jobs=-1,  # cpu's utilizadas
    verbose=3  # mensagens durante o treino
)

# Treino
# Árvore de decisão utiliza dados NÃO padronizados no treinamento
grid_search_ad.fit(atributos_treino_nao_pad, rotulo_treino_nao_pad)

#rótulos preditos pelo teste
rotulo_predito_ad = grid_search_ad.predict(atributos_teste_nao_pad)

# Adicione estas linhas antes da linha que dá erro
print("Formato do rotulo_teste_nao_pad:", rotulo_teste_nao_pad.shape)
print("Formato do rotulo_predito_ad:", rotulo_predito_ad.shape)
print("Tipo dos dados:", type(rotulo_teste_nao_pad))
print("Primeiras linhas do rotulo_teste_nao_pad:")
print(rotulo_teste_nao_pad[:5])
print("Primeiras linhas do rotulo_predito_ad:")
print(rotulo_predito_ad[:5])

# Selecionar apenas uma coluna dos rótulos
rotulo_teste_correto = rotulo_teste_nao_pad.iloc[:, 0]
rotulo_predito_correto = rotulo_predito_ad[:, 0]

# Verificar quantas classes únicas existem
classes_unicas = np.unique(rotulo_teste_correto)
print("Classes únicas nos dados:", classes_unicas)
print("Número de classes:", len(classes_unicas))

# Calcular recall para problema multiclasse
recall_ad = recall_score(rotulo_teste_correto, rotulo_predito_correto, average='weighted') * 100.0
print("Recall (Árvore de Decisão) - weighted:", str(round(recall_ad, 2)) + "%")

# Opcional: calcular outras métricas
recall_macro = recall_score(rotulo_teste_correto, rotulo_predito_correto, average='macro') * 100.0
recall_micro = recall_score(rotulo_teste_correto, rotulo_predito_correto, average='micro') * 100.0

print("Recall (Árvore de Decisão) - macro:", str(round(recall_macro, 2)) + "%")
print("Recall (Árvore de Decisão) - micro:", str(round(recall_micro, 2)) + "%")

# Recall por classe sem average
recall_por_classe = recall_score(rotulo_teste_correto, rotulo_predito_correto, average=None) * 100.0

print("Recall por classe:")
for i, classe in enumerate(classes_unicas):
    print(f"  Classe {classe}: {recall_por_classe[i]:.2f}%")

# Diagrama da Árvore de Decisão
# Árvore de decisão utiliza dados NÃO padronizados no treinamento

plt.figure(figsize=(250, 12))

classes_rotulo  = np.array(grid_search_ad.classes_).astype(str) # Convert list to numpy array before astype
atributos_ad    = grid_search_ad.best_estimator_.feature_names_in_

plot_tree(grid_search_ad.best_estimator_, # melhor modelo selecionado
          feature_names=atributos_ad,     # atributos
          class_names=classes_rotulo,     # classes do rótulo
          filled=True,                    # atribui uma tonalidade diferentes aos nós, quanto mais escura a cor, mais confiante o resultado
          rounded=True,                   # arredonda o visual dos nós
          fontsize=5)                     # tamanho da letra

plt.title('Árvore de decisão', fontsize=16)
plt.show()

"""## 6.Validação verificação da árvore de decisão

### 6.1 matrizes de confusão árvore
"""

# Matriz de confusão da Árvore de Decisão
matrix_confusao_ad = confusion_matrix(rotulo_teste_correto, rotulo_predito_correto)  # gera matriz de confusão

plt.figure(figsize=(4, 3))                            # tamanho do gráfico
sns.heatmap(                                          # sns.heatmap cria mapa de calor
            matrix_confusao_ad,                       # matriz de confusão
            annot=True,                               # valor numérico em cada célula
            fmt='d',                                  # formato(sem casas decimais)
            cmap='Blues',                             # cor das células.
            cbar=False,                               # barra lateral de valores
            linewidths=.5,                            # largura das linhas das células.
            linecolor='black',                        # cor das linhas que separam as células.
            xticklabels=['Saudável', 'Doente'],       # nomes das classes no eixo horizontal
            yticklabels=['Saudável', 'Doente']        # nomes das classes no eixo vertical
            )
plt.title('Matriz de Confusão da Árvore de Decisão')  # título do gráfico
plt.xlabel('Rótulos Preditos')                        # título do eixo horizontal
plt.ylabel('Rótulos Reais')                           # título do eixo vertical
plt.show()

"""### 6.2 métricas de validação"""

print("Métricas de Validação (Árvore de decisão):\n\n", classification_report(rotulo_teste_correto, rotulo_predito_correto))

"""### 6.3 predição de um novo paciente"""

# DADOS DO NOVO PACIENTE
# === ATRIBUTOS NUMÉRICOS (8 features) ===
Idade = 35.0
Nivel_estresse = 6.5
Horas_de_sono = 6.0
Dias_atividade_fisica = 2.0
Depressao = 12.0
Ansiedade = 15.0
Suporte_social = 45.0
Produtividade = 70.0

# === ATRIBUTOS CATEGÓRICOS BINÁRIOS (2 features) ===
Historico_saude_mental = 1.0  # 1 = Sim, 0 = Não
Busca_tratamento = 0.0        # 1 = Sim, 0 = Não

# === ATRIBUTOS CATEGÓRICOS NOMINAIS (10 features) ===
# Gênero: selecione apenas UM como 1.0
Genero_F = 0.0
Genero_M = 1.0  # Masculino
Genero_O = 0.0

# Status emprego: selecione apenas UM como 1.0
Status_empregado_Desempregado = 0.0
Status_empregado_Empregado = 1.0  # Empregado
Status_empregado_Estudante = 0.0
Status_empregado_PJ = 0.0

# Ambiente trabalho: selecione apenas UM como 1.0
Ambiente_de_trabalho_Hibrido = 0.0
Ambiente_de_trabalho_Presencial = 1.0  # Presencial
Ambiente_de_trabalho_Remoto = 0.0

# === CRIANDO O ARRAY ===
novo_paciente = np.array([[
    # 1-8: Atributos numéricos
    Idade, Nivel_estresse, Horas_de_sono, Dias_atividade_fisica,
    Depressao, Ansiedade, Suporte_social, Produtividade,

    # 9-10: Atributos binários
    Historico_saude_mental, Busca_tratamento,

    # 11-13: Gênero
    Genero_F, Genero_M, Genero_O,

    # 14-17: Status emprego
    Status_empregado_Desempregado, Status_empregado_Empregado,
    Status_empregado_Estudante, Status_empregado_PJ,

    # 18-20: Ambiente trabalho
    Ambiente_de_trabalho_Hibrido, Ambiente_de_trabalho_Presencial, Ambiente_de_trabalho_Remoto
]])

print("Formato do novo_paciente:", novo_paciente.shape)
print("Número de features:", novo_paciente.shape[1])
print("Novo paciente foi criado!!!")

# === VERIFICAÇÃO ===
print("\nVerificação da estrutura:")
print("1. Atributos numéricos (8):", novo_paciente[0, :8])
print("2. Atributos binários (2):", novo_paciente[0, 8:10])
print("3. Gênero (3):", novo_paciente[0, 10:13])
print("4. Status emprego (4):", novo_paciente[0, 13:17])
print("5. Ambiente trabalho (3):", novo_paciente[0, 17:20])

# Fazer predição com Árvore de Decisão
predicao_novo_paciente_ad = grid_search_ad.predict(novo_paciente)

print("Formato da predição:", predicao_novo_paciente_ad.shape)
print("Valor bruto da predição:", predicao_novo_paciente_ad)

# Processar a predição (lembrando que tem 2 colunas)
predicao_valor = int(predicao_novo_paciente_ad[0, 0])  # Pega primeira coluna

mapeamento_classes = {
    0: 'Baixo Risco',
    1: 'Médio Risco',
    2: 'Alto Risco'
}

predicao_final_ad = mapeamento_classes.get(predicao_valor, 'Desconhecido')
print(f'Predição (Árvore de decisão) = {predicao_final_ad}')
