# Week 2 Project

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

# Problem 1


# Problem 2
df = pd.read_csv('problem2.csv')

# OLS
y = df.y
x = df.x
X = sm.add_constant(x)  # Adds a constant term to X

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

const, beta = results.params
# e1 = results.resid
e2 = [y[i] - x[i] * beta - const for i in range(0, len(y))]  # Error term = y - x*beta - c

sns.set()
sns.distplot(e2, hist=False, kde=False, fit=stats.gamma)  # hist: display histogram; kde: kernel density
plt.title('PDF of Error Term')
plt.xlabel('e2')
plt.show()

# MLE with normality
# Use the pdf => the likelihood function => derivative => find the minimum
# theta_n = argmin(1/2 sum of (y_i - g(x_i, theta))^2) where g(x_i, theta) = theta_1 * x + theta_0
# Actually the same as OLS
# theta_n,1 = [mean(x*y) - mean(x) * mean(y)]/[mean(x^2) - (mean(x))^2]
# theta_n,0 = mean(y) - theta_1 * mean(x)

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
print("degree of freedom: ", len(x) - 1)  # len(x) = len(y) = 100


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
# y_t = 1.0 + 0.5*y_t-1 + e, e ~ N(0,0.1)
n = 10000
burn_in = 50
y = []
e = np.random.normal(0, 0.1, burn_in + n)

yt_last = 1.0
for i in range(0, n + burn_in):
    temp = 1.0 + 0.5 * yt_last + e[i]
    yt_last = temp
    if i >= burn_in:
        y.append(temp)

sm.graphics.tsa.plot_acf(y, lags=40)
plt.xlim(0.3, 10.3)
plt.ylim(-0.2, 0.6)
plt.xlabel('t')
plt.ylabel('acf')
plt.title('AR(1)')
plt.show()

# AR(2)
# y_t = 1.0 + 0.5*y_t-1 + 0.3*y_t-2 + e, e ~ N(0,0.1)
n = 10000
burn_in = 50
y = []
e = np.random.normal(0, 0.1, burn_in + n)

yt_llast = 1.0
yt_last = 0.5
for i in range(0, n + burn_in):
    temp = 1.0 + 0.5 * yt_last + 0.3 * yt_last + e[i]
    yt_last = temp
    if i >= burn_in:
        y.append(temp)

sm.graphics.tsa.plot_acf(y, lags=20)
plt.xlim(0.3, 10.3)
plt.ylim(-0.2, 0.6)
plt.xlabel('t')
plt.ylabel('acf')
plt.title('AR(1)')
plt.show()
