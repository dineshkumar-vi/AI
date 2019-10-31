import numpy as np
import matplotlib.pyplot as plt

greyHounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyHounds)
lab_height = 24 + 4 * np.random.randn(labs)

# Draw the plot
plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()


