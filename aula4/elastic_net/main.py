import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('day.csv')
df = df.drop(['dteday', 'casual', 'registered'], axis=1)
Y = df['cnt']
X = df.drop('cnt', axis=1)

class ElasticNet:
  alfa = 0.01
  batchSize = 200
  epochs = 20
  w = []
  b = 0
  rho = 0.05

  def train(self, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42)

    rows = X_train.shape[0]
    columns = X_train.shape[1]
    self._init_params(columns)

    for epoch in range(self.epochs):
      trainErr = math.sqrt(self._error(X_train, Y_train))
      testErr = math.sqrt(self._error(X_test, Y_test))
      print(f"Epoch: {epoch}, Train Error: {trainErr}, Test Error: {testErr}")
      
      for i in range(0, rows, self.batchSize):
        batchX = X_train.iloc[i:i+self.batchSize]
        batchY = Y_train.iloc[i:i+self.batchSize]
        self._sgd(batchX, batchY)
    
    trainErr = math.sqrt(self._error(X_train, Y_train))
    testErr = math.sqrt(self._error(X_test, Y_test))
    print(f"Epoch: {self.epochs}, Train Error: {trainErr}, Test Error: {testErr}")

  
  def predict(self, X):
    pred = self.b
    for k in range(len(self.w)):
      pred += self.w[k] * X.iloc[k]
    return pred
    
  def _init_params(self, columns):
    self.w = [random.random() for i in range(columns)]
    self.b = random.random()
  
  def _sgd(self, X, Y):
    (dw, db) = self._derivatives(X, Y)
    for k in range(len(self.w)):
      self.w[k] -= self.alfa * dw[k]
      self.b -= self.alfa * db

  def _derivatives(self, X, Y):
    dw = [2 * self.rho * k for k in self.w]
    db = 0
    N = X.shape[0]

    for i in range(N):
      xi = X.iloc[i]
      yi = Y.iloc[i]
      pred = self.predict(xi)
      diff = pred - yi
      for k in range(len(self.w)):
        dw[k] += diff * xi.iloc[k]
      db += diff
    
    dw = [dw[k] / N for k in range(len(self.w))]
    db = db / N
    return (dw, db)
  
  def l1(self):
    l1 = 0
    for k in range(len(self.w)):
      l1 += abs(self.w[k])
    return l1
  
  def l2(self):
    l2 = 0
    for k in range(len(self.w)):
      l2 += self.w[k] * self.w[k]
    return l2

  def _error(self, X, Y):
    result = 0
    N = X.shape[0]
    for i in range(N):
      xi = X.iloc[i]
      yi = Y.iloc[i]
      err = self.predict(xi) - yi
      result += err * err
    return result / N + self.rho * self.l2()

model = ElasticNet()
model.batchSize = 80
model.epochs = 1000
model.rho = 0.0005
model.alfa = 0.000008
model.train(X, Y)

pX = [model.predict(X.iloc[i]) for i in range(len(X))]
plt.figure(figsize=(10, 6))
plt.plot(X['instant'], Y, color='blue')
plt.plot(X['instant'], pX, color='orange')
plt.show()

err = (Y - pX).abs()
err.hist(bins = 12)