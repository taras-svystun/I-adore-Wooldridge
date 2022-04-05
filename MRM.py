import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

x1 = np.linspace(0, 20, 51)
x2 = 100 - 5 * (x1 - 2 * np.cos(x1))
Beta0, Beta1, Beta2 = 10, 5, 3
y = Beta0 + Beta1 * (x1 + 3 * np.sin(x1)) + Beta2 * (x2 + 12 * np.cos(x2))

fig = px.scatter_3d(x=x1, y=x2, z=y)
# fig.show()

x1_mean = x1.mean()
x2_mean = x2.mean()
y_mean = y.mean()

# beta0 = z_mean - beta1 * x_mean - beta2 * y_mean
df = pd.DataFrame({'x1': x1, 'x2': x2, 'y':y})
X = df[['x1', 'x2']]
Y = df.y

# Estimating y = beta0 + beta1 * x1 + beta2 * x2
s_x1, s_x2 = x1 - x1_mean, x2 - x2_mean
s_y = y - y_mean
s_yy = SST = np.dot(s_y, s_y)


# Regressing x1 on x2: x1 = alpha0 + alpha2 * x2
s_x1x2 = np.dot(s_x1, s_x2)
s_x2x2 = np.dot(s_x2, s_x2)
# or alternatively s_x2x2 = sum((s_x) ** 2) = np.dot(s_x, s_x)
alpha2 = s_x1x2 / s_x2x2
alpha0 = x1_mean - alpha2 * x2_mean
x1_hat = alpha0 + alpha2 * x2
print(f"x1_mean: {x1_mean}, x1_hat.mean: {x1_hat.mean()}.\n\
    If they don't coincide, something is wrong with regression.")

# Residuals on regressing x1 on x2
r1 = x1 - x1_hat
df['r1'] = r1

# Regressing x2 on x1: x2 = gama0 + gama1 * x1
s_x2x1 = s_x1x2
s_x1x1 = np.dot(s_x1, s_x1)
gama1 = s_x2x1 / s_x1x1
gama0 = x2_mean - gama1 * x1_mean
x2_hat = gama0 + gama1 * x1
print(f"x2_mean: {x2_mean}, x2_hat.mean: {x2_hat.mean()}.\n\
    If they don't coincide, something is wrong with regression.")
r2 = x2 - x2_hat
df['r2'] = r2


# Finally, we can estimate beta's, thus regression of y on x1, x2
s_r1, s_r2 = r1, r2
# We can write such an equation, bacause both r1_mean = r2_mean = 0

s_r1y, s_r2y = np.dot(s_r1, s_y), np.dot(s_r2, s_y)
s_r1r1, s_r2r2 = np.dot(s_r1, s_r1), np.dot(s_r2, s_r2)

beta1, beta2 = s_r1y / s_r1r1, s_r2y / s_r2r2
beta0 = y_mean - beta1 * x1_mean - beta2 * x2_mean
y_hat = beta0 + beta1 * x1 + beta2 * x2
print(f"y_mean: {y_mean}, y_hat.mean: {y_hat.mean()}.\n\
    If they don't coincide, something is wrong with regression.")



s_y_hat = y_hat - y_mean
s_y_hat_y_hat = SSE = np.dot(s_y_hat, s_y_hat)
s_r = y_hat - y
s_rr = SSR = np.dot(s_r, s_r)

R2 = SSE / SST
# Or alternatively, R2 = 1 - SSR / SST

model = LinearRegression(fit_intercept=True).fit(X, Y)
beta_0 = model.intercept_
beta_1, beta_2 = model.coef_

result = [
    "Now, it's time to compare my manual linear model with sklearn.",
    "I'll begin with coefficients:",
    "",
    "Manually estimated coeffs:",
    "\n".join([f"intercept: {round(beta0, 4)}", f"slope1: {round(beta1, 4)}", \
    f"slope2: {round(beta2, 4)}"]),
    "",
    "sklearn estimated coeffs:",
    "\n".join([f"intercept: {round(beta_0, 4)}", f"slope1: {round(beta_1, 4)}", \
    f"slope2: {round(beta_2, 4)}"]),
    "",
    "Now, let's compare wheteher R2 coincide",
    f"My R-squared: {round(R2, 4)}",
    f"sklearn R-squared: {round(model.score(X, Y), 4)}"
]

print('\n--------  R E S U L T S  --------')
print('\n'.join(result))
