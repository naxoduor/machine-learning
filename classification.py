from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt


n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)

print(y[:10])

circles = pd.DataFrame({"X0":X[:, 0], "X1": X[:, 1], "label":y})


plt.figure(figsize=(10,7))

plt.scatter( X[:, 0], X[:, 1], c="y", cmap=plt.cm.ColormapRegistry)

plt.legend()

plt.show()


print(circles)