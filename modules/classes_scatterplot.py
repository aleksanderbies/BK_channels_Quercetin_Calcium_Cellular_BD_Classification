import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt

def draw_classes_scatterplot(X, y, title):
    """
    This function draws scatterplot of classified objects
    :param X: Values of time series
    :param y: class of time series
    :param title: title of graph
    """

    Z = np.c_[X, y]
    pred_1 = []
    pred_0 = []

    for i in range(Z.shape[0]):
        if Z[:, -1][i] == 1:
            pred_1.append(Z[i])
        else:
            pred_0.append(Z[i])

    pred_1 = np.array(pred_1)
    pred_0 = np.array(pred_0)

    pred_1 = np.delete(pred_1, -1, axis=1)

    scaler = StandardScaler()
    pca = decomposition.PCA(n_components=3)

    pred_0 = np.delete(pred_0, -1, axis=1)

    scaler.fit(pred_0)

    pred_0 = scaler.transform(pred_0)

    pca.fit(pred_0)

    pred_0 = pca.transform(pred_0)

    scaler.fit(pred_1)

    pred_1 = scaler.transform(pred_1)

    pca.fit(pred_1)

    pred_1 = pca.transform(pred_1)

    plt.figure(figsize=(6, 6))

    ax = plt.axes(projection='3d')

    ax.scatter(pred_1[:, 0], pred_1[:, 1], pred_1[:, 2], alpha=0.8, c="blue", label="No-Que")

    ax.scatter(pred_0[:, 0], pred_0[:, 1], pred_0[:, 2], alpha=0.8, c="red", label="Que")

    plt.title(title)
    plt.legend(loc=2)

    plt.show()