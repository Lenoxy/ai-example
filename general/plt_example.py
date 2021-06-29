import matplotlib.pyplot as plt
import numpy as np

mat = np.random.randint(0, 1000, (200, 200))

plt.imshow(mat, cmap='gist_rainbow')
plt.title('Random Integers visualized')

plt.show()
