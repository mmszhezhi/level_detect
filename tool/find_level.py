import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import UnivariateSpline

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/20.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

img = cv2.imread("edge.png",cv2.COLOR_BGR2GRAY)
sig = np.sum(img,axis=1)
length = len(sig)
sig = sig[int(0.2*length):int(0.8*length)]
sig = sig / 1000
x = np.array(range(sig.shape[0]))

peaks,prop = signal.find_peaks(sig,height=10,threshold=1,distance=50)
plt.plot(sig)
plt.scatter(peaks,prop["peak_heights"])

bands = half_max_x(x[:peaks[0] +20],sig[:peaks[0] +20])
import functools
round2 = functools.partial(round,ndigits=2)
bands = list(map(round2,bands))

print(bands)
plt.scatter(bands,[1,1])

plt.title(f"index of peaks {peaks} height fo peaks {prop['peak_heights']} bands {bands}")
plt.savefig("peaks_bands.png")
plt.show()













