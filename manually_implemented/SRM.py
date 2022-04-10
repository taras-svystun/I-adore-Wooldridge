from matplotlib.pyplot import figure
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# sample size
n = 100 
X = np.linspace(0, 50, n)
y = 5 * (X + 10 * X * np.sin(X)) + 7 * X ** 2
figure = go.Figure(
    go.Scatter(x=X, y=y)
)

x_mean = X.mean()
y_mean = y.mean()
s_x, s_y = X - x_mean, y - y_mean
s_xx = sum((s_x) ** 2)
s_yy = sum((s_y) ** 2)
x_var = s_xx.sum() / n

s_xy = np.dot(s_x, s_y)

# slope and intercept
beta_1 = s_xy / s_xx
beta_0 = y_mean - beta_1 * x_mean

pred_y = beta_0 + beta_1 * X

figure.add_traces(
    go.Scatter(x=X, y=pred_y)
)
# figure.show()

corr = s_xy / np.sqrt(s_xx * s_yy)
print(f'Manual R-squared: {round(corr ** 2, 4)}')

model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
print(f'R-squared from sklearn: {round(model.score(X.reshape(-1, 1), y), 4)}')
