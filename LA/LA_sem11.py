import numpy as np
import pandas as pd
from datetime import datetime
from random import choice, seed
import scipy.linalg as la
import plotly.graph_objects as go
import plotly.express as px

seed(10)

def do_the_job(two_by_two=True):
    if two_by_two:
        A = np.array(((3, 2),
                    (2, 6)))

        eigenvalues, eigenvectors = np.linalg.eig(A)
        print(f'\nEigenvalues\n{eigenvalues}')
        print(f'\nEigenvectors\n{eigenvectors}')

        orth, upper = np.linalg.qr(eigenvectors)
        print(f'\nOrthonormal matrix P\n{orth}')

    else:
        B = np.array(((3, 2, 0),
                      (2, 4, -2),
                      (0, -2, 5)))

        eigenvalues, eigenvectors = np.linalg.eig(B)
        print(f'\nEigenvalues\n{eigenvalues}')
        print(f'\nEigenvectors\n{eigenvectors}')

        orth, upper = np.linalg.qr(eigenvectors)
        print(f'\nOrthonormal matrix P\n{orth}')

# do_the_job(True)

A = np.array(((1, -4, 2),
              (-4, 1, -2),
              (2, -2, -2)))
# print(A.dot(A))
P, L, U = la.lu(A)
# print('P')
# print(P)
# print('L')
# print(L)
# print('U')
# print(U)

# n = 1000
# x = np.linspace(-20, 20, n)
# y1 = (-2 + np.sqrt(x ** 2 + 16)) / 6
# y2 = (-2 + np.sqrt(x ** 2 - 16)) / 6

# figure = go.Figure(go.Scatter(x=x, y=y1))
# figure.add_trace(go.Scatter(x=x, y=y2))


# a = np.linspace(-10, 10, n)
# b1 = np.sqrt(14 - 10 * a ** 2)
# b2 = np.sqrt(14 + 10 * a ** 2)
# x1 = 2 * a + b1
# y1 = -a + 2 * b1

# x2 = 2 * a + b2
# y2 = -a + 2 * b2

# figure = go.Figure(go.Scatter(x=x1, y=y1))
# figure.add_trace(go.Scatter(x=x2, y=y2))
# figure.show()

dates_ = datelist = pd.date_range('2022-01-01', periods=10).tolist()
numbers = list(range(1, 15))
prices = list(range(0, 1000, 5))
dates, category, price = [], [], []
cat = ['фрукти-овочі', 'молочні вироби', 
              'кулінарія', 'солодощі', 'алкоголь']

for date in dates_:
    number = choice(numbers)
    for i in range(number):
        dates.append(date)
        category.append(choice(cat))
        price.append(choice(prices))

data = pd.DataFrame({
    'date': dates,
    'category': category,
    'price': price
})

grouped = data.groupby(['date']).sum().reset_index()
fig = px.line(data_frame=grouped, x='date', y='price',
              text='price', title='Скільки я витратив загалом')
fig.update_traces(textposition="bottom right")
fig.update_yaxes(title_text='грн')

fig.show()