import pandas as pd
from sklearn.preprocessing import LabelEncoder

# limpeza de dados

le = LabelEncoder()

df = pd.read_csv('sample_data/california_housing_train.csv') # abrir o arquivo
df = df.dropna() # removendo pessoas que não tem numero, not a number, tira valores vazios
df = df.fillna("dado ausente") # prenche valores vazios com esse texto
df['housing_median_age'].fillna("não possui idade", inplace = True) # o implace faz com que vc não precise fazer df = df['houst...']
latitudemedia = df['latitude'].mean(); # faz a média de todas as latitudes
df['latitude'] = df.fillna(latitudemedia) # substitui valores vazios com a média das idades

col_of_date = 'housing_median_age'
df[col_of_date] = pd.to_datetime(df[col_of_date])

df.loc(10, 'longitude') = -111.5 # mudando uma linha específica
df.drop(8, inplace = True) # removcendo uam linha


for x in df.index:
    if df.loc(x, 'latitude') < -200:
        df.loc(x, 'latitude') = -114
    elif df.loc(x, 'longitude') > 100:
        df.drop(x, inplace=True)

df.drop_duplicates(inplace = True)

# label encoder

df['tipoSanguineo'] = le.fit_transform(df['tipoSanguineo']) # vai ver todos os tipos e trasnformar em números

#insights : eu devo salvar o label encoder senão nn vou conseguir fazer a tradução dos dados, a qualquer dados novo devemos usar o mesmo label encoder para analisar os dados, deve-se guardar os dados
# textos que o label encoder não conhecem são mais difíceis
# o fit ernsina pro label encoder como ele deve codificar o dado

df # mostra o arquivo