import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ============ LIMPEZA DE DADOS ==================

le = LabelEncoder()

df = pd.read_csv('sample_data/california_housing_train.csv') # abrir o arquivo
df = df.dropna() # removendo pessoas que não tem numero, not a number, tira valores vazios
df = df.fillna("dado ausente") # prenche valores vazios com esse texto
df['housing_median_age'].fillna("não possui idade", inplace = True) # o inplace faz com que vc não precise fazer df = df['houst...']
latitudemedia = df['latitude'].mean(); # faz a média de todas as latitudes
df['latitude'] = df.fillna(latitudemedia) # substitui valores vazios com a média das latitudes

col_of_date = 'housing_median_age'
df[col_of_date] = pd.to_datetime(df[col_of_date])

df.loc(10, 'longitude') = -111.5 # mudando uma linha específica
df.drop(8, inplace = True) # removendo uam linha

df # mostra o arquivo


for x in df.index:
    if df.loc(x, 'latitude') < -200:
        df.loc(x, 'latitude') = -114
    elif df.loc(x, 'longitude') > 100:
        df.drop(x, inplace=True)

df.drop_duplicates(inplace = True)

# label encoder
df['tipoSanguineo'] = le.fit_transform(df['tipoSanguineo']) # vai ver todos os tipos e transformar em números

# ============ CONSTRUINDO GRÁFICOS (Análise Exploratória) ==============================

# !pip install matplotlib

# ============== HISTOGRAMA #

# df = pd.read_csv('sample_data/california_housing_train.csv') # abrir o arquivo

plt.hist(df['latitude'])
plt.xlabel('quantidade de casas')
plt.ylabel('valor da casa')


# ============== BOXPLOT #

# mostra mínimo e máximo, dividido em valores de 25% e deixa outliers fora, linha verde é a mediana : 50% pra baixo e 50% pra cima(lembrando dos pontos fora da curva); 
df.boxplot(column = ['latitude'])

# ============== SCATTER PLOT #

df.plot.scatter(x = 'latitude', y = 'longitude', s = 1)

# ==============  #


# =============================== ANOTAÇÕES


#insights : eu devo salvar o label encoder senão nn vou conseguir fazer a tradução dos dados, a qualquer dados novo devemos usar o mesmo label encoder para analisar os dados, deve-se guardar os dados
# textos que o label encoder não conhecem são mais difíceis
# o fit ensina pro label encoder como ele deve codificar o dado
# graficos lienares mostgram que duas informações talvez não sejam tão importantes, se houver relação é possível ficar apenas com uma

"""



# season (1:springer, 2:summer, 3:fall, 4:winter)
# demorou 10 anos para voltarem a emprestar bikes?

# df.plot.scatter(x = 'hr', y = 'cnt', s = 1)

# quanto mais humido mais bikes
# quanto mais vento menos bikes
# quanto mais a estação é fria mais bikes



# for i in df['season'] :
#   if i != 1 :
#     print(i)

# df
# # obtendo o nosso objetivo
# Y = df['median_house_value']
# # removendo o objetivo do resto dos dados
# X = df.drop('median_house_value', axis=1)
# # test_size = proporção que vai para teste
# # random_state = semente aleatória para embaralhar os dados
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# model = DecisionTreeClassifier(
#     criterion = "entropy",
#     max_depth = 20,
#     min_samples_split = 10
# )

# model.fit(X_train, Y_train)

# model = DecisionTreeRegressor(
#     max_depth = 20,
#     min_samples_split = 10
# )

# # Criando e treinando o modelo
# # model = DecisionTreeRegressor()
# model.fit(X_train, Y_train)
# # salva o modelo treinado para uso posterior
# dump(model, 'filename.joblib')
# # model = load('filename.joblib') # carrega o modelo, evidentemente não é necessário

# Y_real = Y_train
# Y_pred = model.predict(X_train)
# train_error = mean_absolute_error(Y_real, Y_pred)
# Y_real = Y_test
# Y_pred = model.predict(X_test)
# test_error = mean_absolute_error(Y_real, Y_pred)
# print(train_error, test_error) # 50786 50922

# # ele printa o erro, nesse caso eu errei 16 mil dolares
# vendo cada item das de weekday das 10 primerias colunas
# for i in primeiros_10:
#     if 'weekday' in i:
#         print(i['weekday'])
"""