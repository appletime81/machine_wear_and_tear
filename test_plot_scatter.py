import numpy as np
import matplotlib.pyplot as plt


x = [i for i in range(100)]
y = [i for i in range(100)]
color = []
for i in range(50):
    color.append((255/255, 0, 0))
for i in range(50):
    color.append((0, 255/255, 0))
plt.scatter(x, y, c=color)
plt.show()



