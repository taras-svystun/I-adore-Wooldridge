import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

print('------------------ Problem 5a) ------------------')
A = np.array(((1, -1),
              (2, 3),
              (4, 5)))

b = np.array((2, -1, 5))
inner = A.transpose().dot(A)
print(f"A-trans * A * x0 = A-trans * b")
print()
print(f"{inner}, A-trans * A")
print()
print(f"{A.transpose()}, A-trans")

inv = np.linalg.inv(inner)
x0 = inv.dot(A.transpose())
# x0 = (AT * A)^-1 * AT * b
x0 = x0.dot(b)
print()
print(f"Least squares solution = {x0}")

projection = A.dot(x0)
# print(projection)
error = b - projection
print()
print(error)
print(f"It's length = {np.linalg.norm(error)}")
print()

print('------------------ Problem 5b) ------------------')
A = np.array(((2, -2),
              (1, 1),
              (1, 3)))
b = np.array((7, 0, -7))

inner = A.transpose().dot(A)
print(f"{A.transpose()}, A-trans")
print()
print(f"A-trans * A * x0 = A-trans * b")
print()
print(f"{inner}, A-trans * A")
inv = np.linalg.inv(inner)
x0 = inv.dot(A.transpose())
x0 = x0.dot(b)
print()
print(f"Least squares solution = {x0}")

error = b - A.dot(x0)
print()
print(error)
print(f"It's length = {np.linalg.norm(error)}")
print()

print('------------------ Problem 6a) ------------------')
X = np.array(((1, 0),
              (1, 1),
              (1, 2)))

Y = np.array((1, 2, 7))

inner = X.transpose().dot(X)
inv = np.linalg.inv(inner)
coeffs = inv.dot(X.transpose())
coeffs = coeffs.dot(Y)
print(f"Coefficients are {coeffs}")
fit = X.dot(coeffs)

fig1 = px.scatter(x=[0, 1, 2], y=Y)
fig1.update_traces(marker=dict(size=12, color='red',
                              line=dict(width=2,
                                        color='Green')),
                  selector=dict(mode='markers'))

fig2 = px.line(x=[0, 1, 2], y=fit)
fig2.update_traces(line=dict(dash='dash'))
figure = go.Figure(data=fig1.data + fig2.data)
figure.update_layout(
    title='Fitting the line')
# figure.show()

print()
print('------------------ Problem 6b) ------------------')
x = np.array([2, 3, 5, 6])
Y = np.array([0, -10, -48, -76])
data = pd.DataFrame({'y' : Y,
                     '0' : [1, 1, 1, 1],
                     '1' : x,
                     '2' : x ** 2})
# print(data)

X = data[['0', '1', '2']]
inner = X.transpose().dot(X)
inv = np.linalg.inv(inner)
coeffs = inv.dot(X.transpose())
coeffs = coeffs.dot(Y)
print(f"Coefficients are {coeffs}")
# y = 2 + 5x - 3x^2
fit = X.dot(coeffs)

fig1 = px.scatter(x=x, y=Y)
fig1.update_traces(marker=dict(size=12, color='red',
                              line=dict(width=2,
                                        color='Green')),
                  selector=dict(mode='markers'))

fig2 = px.line(x=x, y=fit)
fig2.update_traces(line=dict(dash='dash'))
figure = go.Figure(data=fig1.data + fig2.data)
figure.update_layout(
    title='Fitting the quadratic polynomial')
# figure.show()


print()
print('------------------ Problem 6c) ------------------')
x = np.array([2, 3, 5, 6])
Y = np.array([0, -10, -48, -76])
data = pd.DataFrame({'y' : Y,
                     '0' : [1, 1, 1, 1],
                     '1' : x,
                     '2' : x ** 2,
                     '3' : x ** 3})
# print(data)

X = data[['0', '1', '2', '3']]
inner = X.transpose().dot(X)
inv = np.linalg.inv(inner)
coeffs = inv.dot(X.transpose())
coeffs = coeffs.dot(Y)
print(f"Coefficients are {coeffs}")
# y = 2 + 5x - 3x^2 + 0x^3
fit = X.dot(coeffs)

fig1 = px.scatter(x=x, y=Y)
fig1.update_traces(marker=dict(size=12, color='red',
                              line=dict(width=2,
                                        color='Green')),
                  selector=dict(mode='markers'))

fig2 = px.line(x=x, y=fit)
fig2.update_traces(line=dict(dash='dash'))
figure = go.Figure(data=fig1.data + fig2.data)
figure.update_layout(
    title='Fitting the cubic polynomial')
# figure.show()
