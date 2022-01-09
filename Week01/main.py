# Week 01
from typing import List, Any

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statistics import mean, variance
import math
import random

# PDF Example
# Method 1: Generate 10000 standard normal random variables
mu, sigma, size = 0, 1, 100000
X = np.random.normal(mu, sigma, size)

# plot frequency histogram
bins = 100
n, bins, patches = plt.hist(X, bins, facecolor='blue', edgecolor='black')
plt.title('Frequency histogram of X')
plt.xlabel('X')
plt.ylabel('Frequency')
plt.show()

# plot probability histogram
plt.hist(X, bins, facecolor='blue', edgecolor='black', density='true')
plt.title('Probability histogram of X')
plt.xlabel('X')
plt.ylabel('Probability')
plt.show()

# fit histogram curve
sns.set()
sns.distplot(X, hist=False, kde=False, fit=stats.gamma)  # hist: display histogram; kde: kernel density
plt.title('PDF')
plt.xlabel('X')
plt.ylim(-0.003, 0.42)
plt.xlim(-5, 5)
plt.show()

# Method 2: plot standard normal pdf
x = np.linspace(-5, 5, 100000)
y = stats.norm.pdf(x, 0, 1)
plt.plot(x, y, c='b')
plt.xlabel('X')
plt.title('PDF')
plt.savefig("pdf.png")
plt.show()

# CDF
yy = stats.norm.cdf(x, 0, 1)
plt.plot(x, yy, c='b')
plt.title('CDF')
plt.xlabel('X')
plt.savefig("cdf.png")
plt.show()


# calculation of moments

# simulate based on the defined Distribution above, N(0,1)
# Expect  μ = 0,
#       σ^2 = 1,
#      skew = 0,
#      kurt = 3 (excess = 0)

def first4moments(sample):
    N = len(sample)
    print("N: ", N)

    # mean
    _sum = 0
    for i in range(0, N):
        _sum += sample[i]
    μ_hat = _sum / N

    # remove the mean from the sample
    sim_corrected = [sample[i] - μ_hat for i in range(0, N)]
    cm2 = sum([pow(sim_corrected[i], 2) for i in range(0, N)]) / N  # biased

    # variance
    σ2_hat = sum([pow(sim_corrected[i], 2) for i in range(0, N)]) / (N - 1)

    # skew
    skew_hat = sum([pow(sim_corrected[i], 3) for i in range(0, N)]) / N / math.sqrt(cm2 * cm2 * cm2)
    # biased                                                         here, normalizing

    # kurtosis
    kurt_hat = sum([pow(sim_corrected[i], 4) for i in range(0, N)]) / N / (cm2 * cm2)

    excessKurt_hat = kurt_hat - 3
    return μ_hat, σ2_hat, skew_hat, excessKurt_hat


# np.random.seed(40)
d = np.random.normal(0, 1, 10000)
nn = 5000
sim = random.sample(list(d), nn)
m, s2, sk, k = first4moments(sim)
s = pd.Series(sim)
mean_diff = m - s.mean()  # 调用公式算mean/std时， 转为series
var_diff = s2 - s.var()
skew_diff = sk - stats.skew(sim)
kurt_diff = k - stats.kurtosis(sim)

print("Mean difference = ", mean_diff)
print("Variance difference = ", var_diff)
print("Skewness difference = ", skew_diff)
print("Kurtosis difference = ", kurt_diff)

# Test the kurtosis function for bias in small sample sizes
kurt = []
times = 100000
for i in range(0, times):
    dd = np.random.normal(0, 1, 100)
    kurt.append(stats.kurtosis(dd))

kurt_df = pd.DataFrame(kurt)
print(kurt_df.describe())

t = kurt_df.mean() / math.sqrt(kurt_df.var() / times)
p = 2 * (1 - stats.t.cdf(abs(t), times - 1))
# p-value：1-scipy.stats.t.cdf(abs(chi2_score),df)---left tail or right tail，two tails: need to multiple 2
print("t: ", t)
print("p-value: ", p)

# using the Included TTest
tt, p2 = stats.ttest_1samp(kurt_df, 0)
print("tt: ", tt)
print("p2: ", p2)
