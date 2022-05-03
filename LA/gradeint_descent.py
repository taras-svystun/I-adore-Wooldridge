import plotly.express as px
import plotly.graph_objects as go
import numpy as np

dx = 0.001
LEARNING_RATE = 0.0001

def cost(actual, predicted):
    return ((actual - predicted) ** 2).sum()

def predict(i, s):
    return i + s * x

# dcost / ds
# cost = ((actual - predicted(s, i)) ** 2).sum()
# 

def grad(x, y, i, s):
    return (
        (-2 * (y - (i + s * x))).sum(),
        (-2 * (y - (i + s * x)) * x).sum()
    )

x = np.linspace(0, 20, 21)
y = 10 + 3 * x

i, s = 0, 0

for _ in range(5000):
    dcdi, dcds = grad(x, y, i, s)
    i -= dcdi * LEARNING_RATE
    s -= dcds * LEARNING_RATE 


fig = go.Figure()
fig.add_trace(go.Line(x=x, y=y, mode='lines+markers', name='Actual',
                      line_width=3, marker={'size': 10}))
fig.add_trace(go.Line(x=x, y=(i + s * x),
                      mode='lines+markers', name='Predicted',
                      line_width=3, marker={'size': 10}))
for real_x, real_z in zip(x, y):
    fig.add_trace(go.Line(x=[real_x]*20,
    y=np.linspace(i + s * real_x, real_z, 20),
    line_color='red', line_dash='dash', line_width=1))
fig.show()