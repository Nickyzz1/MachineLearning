import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

df = pd.read_csv('day.csv')

df.drop(
['casual', 'registered', 'dteday', 'yr', 'mnth'],
axis = 1,
inplace = True
)

Y = df['cnt']
X = df.drop('cnt', axis = 1)

scores = cross_val_score(ElasticNet(fit_intercept = True), X, Y, cv = 8)
print(scores)
pca = PCA(n_components = 8)
pca.fit(X)
X = pca.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15,
random_state=42)

model = GridSearchCV (
ElasticNet(fit_intercept = True),
{
    'alpha': list(map(lambda x: x / 10, range(1, 10))), # ele está treinando 100 modelos
    'l1_ratio': list(map(lambda x: x / 10, range(1, 10))),
}, n_jobs = 4 )

# quando mdoelos mais baiscos nn funcionárem ir para SVM
 
# model = GridSearchCV(SVR(degree = 10) # nn é o elastic net
#    'kernel' : ['linear','rbf', 'poly']
#  ))


model.fit(X_train, Y_train)
print(model.best_params_)
model = model.best_estimator_

dump(model, 'model.pkl') # salva o resultado do modelo em um arquivo
print(mean_absolute_error(Y, model.predict(X)))

Ypred = model.predict(X)
plt.plot(Y)
plt.plot(Ypred)
plt.show()
wR = []
wP = []
Ymm = []
Ypmm = []
for i in range(len(Y)):
  wR.append(Y[i])
  wP.append(Ypred[i])
  if len(wR) > 15:
    Ymm.append(sum(wR) / 15)
    Ypmm.append(sum(wP) / 15)
    wR.pop(0)
    wP.pop(0)

plt.plot(Ymm)
plt.plot(Ypmm)
plt.show()