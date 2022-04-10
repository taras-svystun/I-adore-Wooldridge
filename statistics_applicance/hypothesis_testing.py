import pandas as pd
import numpy as np
from random import randint, seed
from math import sqrt

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from prepare_data import prepare_data


data = prepare_data()
# print(data.head())
data = sm.add_constant(data)
seed(1)
data['random'] = [randint(0, 100) for _ in range(data.shape[0])]

features = ['random', 'gender', 'race', 'pareduc', 'lunch', 'prepar']
X = data[features]
y, reading, writing = data['math'], data['reading'], data['writing']

# sklearn model
sk_model = LinearRegression(fit_intercept=True).fit(X=X, y=y)
sk_R2 = sk_model.score(X, y)
sk_coeffs = [sk_model.intercept_] + list(sk_model.coef_)

# statsmodels model
X = data[['const'] + features]
sm_model = sm.OLS(y, X)
fitted = sm_model.fit()
# print(fitted.summary())

''' Now let's test some hypothesis '''

n, temp = data.shape[0], data.copy()
temp = temp[['const', 'random', 'prepar', 'math']]

c = temp['const']
r = temp['random']
p = temp['prepar']

y = temp['math']
test = ['const', 'random', 'prepar']
X = temp[test]
model = sm.OLS(y, X)
fitted = model.fit()
# print(fitted.summary())

'''I'll start with t and f-statistics for 2 control variables: random and prepar'''
random_coef, prepar_coef = fitted.params[1], fitted.params[-1]
t_random, t_prepar = fitted.tvalues[1], fitted.tvalues[-1]

# Manually
y_pred = fitted.predict(X)
temp['y_pred'] = y_pred
temp['residuals'] = y_pred - y

s_uu = sum(temp['residuals'] ** 2)
var_u = s_uu / n
sd_u = sqrt(var_u)

s_pp = sum((p - p.mean()) ** 2)
var_p = s_pp / n
sd_p = sqrt(var_p)

s_rr = sum((r - r.mean()) ** 2)
var_r = s_rr / n
sd_r = sqrt(var_r)

helper_pr = sm.OLS(p, r)
fitted_pr = helper_pr.fit()
R2_pr = fitted_pr.rsquared
se_p = sd_u / sqrt(s_pp * (1 - R2_pr))
t_value_p = prepar_coef / se_p
# print(fitted.summary())
# print(t_value_p)

f = fitted.fvalue
print(f)

'''Standartization'''
print(temp.head())
temp['r_norm'] = (r - r.mean()) / np.std(r)
temp['p_norm'] = (p - p.mean()) / np.std(p)
y_norm = temp['y_norm'] = (y - y.mean()) / np.std(y)

norm_model = sm.OLS(y_norm, temp[['r_norm', 'p_norm']])