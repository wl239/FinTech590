# Week 2 Project
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from scipy import stats


# Problem 1
def build_ols_model(data):
    Y = data.y
    x_i = data.x
    X_i = sm.add_constant(x_i)

    ols = sm.OLS(Y, X_i)
    result = ols.fit()
    summary = result.summary()
    c, slope = result.params
    error_term = result.resid
    return summary, c, slope, error_term


data = pd.read_csv('problem1.csv')

# Multivariate Distribution
# E(Y|X = x) = μ_y + cov_xy/σ2_x * (x - μ_x)
μ_x = sum(data.x)/len(data.x)
σ2_x = sum([(x - μ_x) ** 2 for x in data.x]) / (len(data.x) - 1)
μ_y = sum(data.y)/len(data.y)
σ2_y = sum([(y - μ_y) ** 2 for y in data.y]) / (len(data.y) - 1)
X = data.x
Y = data.y
cov_xy = sum([(X[i] - μ_x) * (Y[i] - μ_y) for i in range(len(X))]) / (len(X) - 1)
beta = cov_xy/σ2_x
print("beta: ", beta)

# OLS
summary, c, slope, error_term = build_ols_model(data)
print("slope: ", slope)

print("diff: ", beta - slope)  # 0


# Problem 2
df = pd.read_csv('problem2.csv')

# OLS
summary_info, const, beta, resid = build_ols_model(df)
print(summary_info)

# e1 = results.resid
# e2 = [y[i] - x[i] * beta - const for i in range(0, len(y))]  # Error term = y - x*beta - c

sns.set()
sns.distplot(resid, hist=False, kde=False, fit=stats.gamma)  # hist: display histogram; kde: kernel density
plt.title('PDF of Error Term')
plt.xlabel('residual')
plt.show()

# MLE with normality
# Use the pdf => the likelihood function => derivative => find the minimum
# theta_n = argmin(1/2 sum of (y_i - g(x_i, theta))^2) where g(x_i, theta) = theta_1 * x + theta_0
# Actually the same as OLS
# theta_n,1 = [mean(x*y) - mean(x) * mean(y)]/[mean(x^2) - (mean(x))^2]
# theta_n,0 = mean(y) - theta_1 * mean(x)

x = df.x
y = df.y
temp = [x[i] * y[i] for i in range(0, len(x))]
m_xy = np.mean(temp)  # mean(x*y)
m_x = np.mean(x)  # mean(x)
m_y = np.mean(y)  # mean(y)
temp = [x[i] * x[i] for i in range(0, len(x))]
m_x2 = np.mean(temp)  # mean(x^2)

theta1 = (m_xy - m_x * m_y) / (m_x2 - m_x * m_x)
theta0 = m_y - theta1 * m_x

TSS = np.sum([(y[i] - m_y) * (y[i] - m_y) for i in range(0, len(y))])
ESS = np.sum([(y[i] - theta1 * x[i] - theta0) ** 2 for i in range(0, len(y))])
R_square = 1 - ESS / TSS

print("theta 1: ", theta1)
print("theta 0: ", theta0)
print("R_square: ", R_square)

# MLE with T distribution
# Use T distribution pdf => derivative: two complex function => gradient descend
# print("degree of freedom: ", len(x) - 1)  # len(x) = len(y) = 100


def compute_error_for_line_given_points(b, m, df):
    totalError = 0
    for i in range(0, len(df)):
        x_i = df.x[i]
        y_i = df.y[i]
        totalError += (y_i - (m * x_i + b)) ** 2
    return totalError / float(len(df))


def compute_r_square(b, m, df):
    TSS = 0
    ESS = 0
    y_mean = np.mean(df.y)
    for i in range(0, len(df)):
        x_i = df.x[i]
        y_i = df.y[i]
        ESS += (y_i - (m * x_i + b)) ** 2
        TSS += (y_i - y_mean) ** 2
    R_square = 1 - ESS / TSS
    return R_square


def step_gradient(b_current, m_current, df, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(df))
    for i in range(0, len(df)):
        x_i = df.x[i]
        y_i = df.y[i]
        b_gradient += -2 * (y_i - b_current - m_current * x_i) / (N - 1 + (y_i - b_current - m_current * x_i) ** 2)
        m_gradient += -2 * x_i * (y_i - b_current - m_current * x_i) / (
                N - 1 + (y_i - b_current - m_current * x_i) ** 2)
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(df, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, df, learning_rate)
    return [b, m]


def run():
    learning_rate = 0.001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, R^2 = {2}".format(initial_b, initial_m,
                                                                            compute_r_square(
                                                                                initial_b, initial_m, df)))
    print("Running...")
    [b, m] = gradient_descent_runner(df, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, R^2 = {3}".format(num_iterations, b, m,
                                                                    compute_r_square(b, m,
                                                                                     df)))


run()
# then compare the R_Square of two models


# Problem 3

# AR(1)
# y_t = 1.0 + 0.5*y_t-1 + e, e ~ N(0,0.01)
n = 10000
burn_in = 50
y1 = []
e = np.random.normal(0, 0.1, burn_in + n)

yt_last = 1
for i in range(0, n + burn_in):
    temp = 1.0 + 0.5 * yt_last + e[i]
    yt_last = temp
    if i >= burn_in:
        y1.append(temp)

# AR(2)
# y_t = 1.0 + 0.5*y_t-1 + 0.3*y_t-2 + e, e ~ N(0,0.01)
y2 = []
# e = np.random.normal(0, 0.1, burn_in + n)

yt_llast = 1
yt_last = 0.5
for i in range(0, n + burn_in):
    temp = 1.0 + 0.5 * yt_last + 0.3 * yt_llast + e[i]
    yt_llast = yt_last
    yt_last = temp
    if i >= burn_in:
        y2.append(temp)

# AR(3)
# y_t = 1.0 + 0.5*y_t-1 + 0.3*y_t-2 + + 0.1*y_t-3 + e, e ~ N(0,0.01)
y3 = []
# e = np.random.normal(0, 0.1, burn_in + n)

yt_lllast = 1
yt_llast = 0.5
yt_last = 0.3
for i in range(0, n + burn_in):
    temp = 1.0 + 0.5 * yt_last + 0.3 * yt_llast + 0.1 * yt_lllast + e[i]
    yt_lllast = yt_llast
    yt_llast = yt_last
    yt_last = temp
    if i >= burn_in:
        y3.append(temp)

fig, ax = plt.subplots(3, 2, figsize=(12, 15))
sm.graphics.tsa.plot_acf(y1, lags=10, ax=ax[0, 0])
sm.graphics.tsa.plot_pacf(y1, lags=10, ax=ax[0, 1], method='ywm')
sm.graphics.tsa.plot_acf(y2, lags=10, ax=ax[1, 0])
sm.graphics.tsa.plot_pacf(y2, lags=10, ax=ax[1, 1], method='ywm')
sm.graphics.tsa.plot_acf(y3, lags=10, ax=ax[2, 0])
sm.graphics.tsa.plot_pacf(y3, lags=10, ax=ax[2, 1], method='ywm')

ax[0, 0].set(ylabel='AR(1)')
ax[1, 0].set(ylabel='AR(2)')
ax[2, 0].set(ylabel='AR(3)')

for ax in ax.flat:
    ax.set_ylim([-0.2, 1])
    ax.set_xlim([0.3, 11])
    ax.set(xlabel='t')
plt.savefig("ACF and PACF of AR.png")
plt.show()

# MA1
# y_t = 1.0 + 0.5*e_t-1 + e, e ~ N(0,.01)
n = 10000
burn_in = 50
y1 = []
e = np.random.normal(0, 0.1, burn_in + n)

for i in range(1, n + burn_in):
    y_t = 1.0 + 0.5 * e[i - 1] + e[i]
    if i >= burn_in:
        y1.append(y_t)

# MA2
# y_t = 1.0 + 0.5*e_t-1 + 0.3*e_t-2 + e, e ~ N(0,.01)
y2 = []

for i in range(2, n + burn_in):
    y_t = 1.0 + 0.5 * e[i - 1] + 0.3 * e[i - 2] + e[i]
    if i >= burn_in:
        y2.append(y_t)

# MA2
# y_t = 1.0 + 0.5*e_t-1 + 0.3*e_t-2 + 0.1*e_t-3 + e, e ~ N(0,.01)
y3 = []

for i in range(3, n + burn_in):
    y_t = 1.0 + 0.5 * e[i - 1] + 0.3 * e[i - 2] + 0.1 * e[i - 3] + e[i]
    if i >= burn_in:
        y3.append(y_t)

fig, ax = plt.subplots(3, 2, figsize=(12, 15))
sm.graphics.tsa.plot_acf(y1, lags=10, ax=ax[0, 0])
sm.graphics.tsa.plot_pacf(y1, lags=10, ax=ax[0, 1], method='ywm')
sm.graphics.tsa.plot_acf(y2, lags=10, ax=ax[1, 0])
sm.graphics.tsa.plot_pacf(y2, lags=10, ax=ax[1, 1], method='ywm')
sm.graphics.tsa.plot_acf(y3, lags=10, ax=ax[2, 0])
sm.graphics.tsa.plot_pacf(y3, lags=10, ax=ax[2, 1], method='ywm')

ax[0, 0].set(ylabel='MA(1)')
ax[1, 0].set(ylabel='MA(2)')
ax[2, 0].set(ylabel='MA(3)')

for ax in ax.flat:
    ax.set_ylim([-0.2, 1])
    ax.set_xlim([0.3, 11])
    ax.set(xlabel='t')
plt.savefig("ACF and PACF of MA.png")
plt.show()
