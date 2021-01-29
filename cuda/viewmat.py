# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot([1,2,3])
im = np.loadtxt('out.txt')
ax.imshow(im, cmap='gray')
fig.savefig('out.png')
