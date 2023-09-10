import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

# define x and y values to use for CDF
f, (ax1, ax2) = plt.subplots(1, 2)
x = np.linspace(-5, 5, 1000)
norm_cdf = ss.norm.cdf(x, loc=0, scale=2.0)
norm_pdf = np.gradient(norm_cdf, np.tanh(x))
ax1.plot(np.tanh(x), norm_cdf, label='x~N(0, 2.0)')
ax2.plot(np.tanh(x), norm_pdf, label='x~N(0, 2.0)')
# clip_cdf = copy.deepcopy(norm_cdf)
# clip_cdf[x < -1] = 0
# clip_cdf[x > 1] = 1
# plt.plot(x, clip_cdf)
norm_cdf = ss.norm.cdf(x, loc=0, scale=1.0)
norm_pdf = np.gradient(norm_cdf, np.tanh(x))
ax1.plot(np.tanh(x), norm_cdf, label='x~N(0, 1.0)')
ax2.plot(np.tanh(x), norm_pdf, label='x~N(0, 1.0)')
# clip_cdf = copy.deepcopy(norm_cdf)
# clip_cdf[x < -1] = 0
# clip_cdf[x > 1] = 1
# plt.plot(x, clip_cdf)
norm_cdf = ss.norm.cdf(x, loc=0, scale=0.5)
norm_pdf = np.gradient(norm_cdf, np.tanh(x))
ax1.plot(np.tanh(x), norm_cdf, label='x~N(0, 0.5)')
ax2.plot(np.tanh(x), norm_pdf, label='x~N(0, 0.5)')
# clip_cdf = copy.deepcopy(norm_cdf)
# clip_cdf[x < -1] = 0
# clip_cdf[x > 1] = 1
# plt.plot(x, clip_cdf)
norm_cdf = ss.norm.cdf(x, loc=0, scale=0.1)
norm_pdf = np.gradient(norm_cdf, np.tanh(x))
ax1.plot(np.tanh(x), norm_cdf, label='x~N(0, 0.1)')
ax2.plot(np.tanh(x), norm_pdf, label='x~N(0, 0.1)')
# clip_cdf = copy.deepcopy(norm_cdf)
# clip_cdf[x < -1] = 0
# clip_cdf[x > 1] = 1
# plt.plot(x, clip_cdf)
ax1.set_xlabel('u=tanh(x)')
ax1.set_ylabel("CDF(u)")
ax1.legend()
ax2.set_xlabel('u=tanh(x)')
ax2.set_ylabel("PDF(u)")
ax2.set_ylim([0, 5])
# ax2.set_yscale('log')
ax2.legend()
plt.show()
# norm_pdf = ss.norm.pdf(x, loc=0, scale=1)
# plot normal CDF
# plt.plot(x, norm_pdf)
# idx1, idx2 = np.argmin((x + 1) ** 2), np.argmin((x - 1) ** 2)
# clip_pdf = norm_pdf[idx1:idx2]
# clip_pdf[0] = sum(norm_pdf[:idx1])
# clip_pdf[-1] = sum(norm_pdf[idx2:])
# plt.plot(x[idx1:idx2], clip_pdf)
# plt.show()
# y = ss.beta.pdf(x, a=0.95, b=1.07)
# y1 = np.exp(x)
# y2 = np.log(1 + np.exp(x))
# y3 = x
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.show()
pass
