import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

mu = 3.0
# define x and y values to use for CDF
f, (ax1, ax2) = plt.subplots(1, 2)
x = np.linspace(-5, 5, 1000)

for std in [2.0, 1.0, 0.7, 0.5, 0.2, 0.1]:
    cdf = ss.norm.cdf(x, loc=mu, scale=std)
    # cdf[x < -1] = 0
    # cdf[x > 1] = 1
    # pdf = np.gradient(cdf, x)
    pdf = np.gradient(cdf, np.tanh(x))
    ax1.plot(np.tanh(x), cdf, label=f'x~N({mu}, {std})')
    ax2.plot(np.tanh(x), pdf, label=f'x~N({mu}, {std})')

# ax1.set_xlabel('x')
# ax1.set_ylabel("CDF(x)")
ax1.set_xlabel('u=tanh(x)')
ax1.set_ylabel("CDF(u)")
ax1.legend()
# ax2.set_xlabel('x')
# ax2.set_ylabel("PDF(x)")
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
