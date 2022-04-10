from prepare_data import prepare_data
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from random import randint, seed

data = prepare_data()
data = (data - data.mean()) / data.std()
data = sm.add_constant(data)
seed(1)
data['random'] = [randint(0, 100) for _ in range(data.shape[0])]

features = 'const$random$gender$race$pareduc$lunch$prepar'.split('$')
X = data[features]
y = data['math']

model = sm.OLS(y, X)
fitted = model.fit()
print(fitted.summary())