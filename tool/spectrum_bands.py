from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

# Sample data with two peaks: small one at t=0.4, large one at t=0.8
ts = np.arange(0, 1, 0.01)
xs = np.exp(-((ts-0.4)/0.1)**2) + 2*np.exp(-((ts-0.8)/0.1)**2)

# Say we have an approximate starting point of 0.35
start_point = 0.35

# Nearest index in "ts" to this starting point is...
start_index = np.argmin(np.abs(ts - start_point))

# Find the local maxima in our data by looking for a sign change in
# the first difference
# From http://stackoverflow.com/a/9667121/188535
maxes = (np.diff(np.sign(np.diff(xs))) < 0).nonzero()[0] + 1

# Find which of these peaks is closest to our starting point
index_of_peak = maxes[np.argmin(np.abs(maxes - start_index))]

print("Peak centre at: %.3f" % ts[index_of_peak])

# Quick plot showing the results: blue line is data, green dot is
# starting point, red dot is peak location
plt.plot(ts, xs, '-b')
plt.plot(ts[start_index], xs[start_index], 'og')
plt.plot(ts[index_of_peak], xs[index_of_peak], 'or')
plt.show()