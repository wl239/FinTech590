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


# MLE with T distribution


# Problem 3

# AR(1)
# y_t = 1.0 + 0.5*y_t-1 + e, e ~ N(0,0.1)
n = 10000
burn_in = 50
y = []
e = np.random.normal(0, 0.1, burn_in+n)

yt_last = 1.0
for i in range(0, n+burn_in):
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
e = np.random.normal(0, 0.1, burn_in+n)

yt_llast = 1.0
yt_last = 0.5
for i in range(0, n+burn_in):
    temp = 1.0 + 0.5 * yt_last + 0.3*yt_last + e[i]
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



