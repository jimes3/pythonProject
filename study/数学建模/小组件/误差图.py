import numpy as np
import matplotlib.pyplot as plt


x = np.arange(5)
y = (25, 32, 34, 20, 25)
y_offset = (3, 5, 2, 3, 3)
plt.errorbar(x, y, yerr=y_offset, capsize=3, capthick=2,ecolor='k',elinewidth=1,
             mec='k',mew=1,ms=10,alpha=1,label="Observation")
plt.show()