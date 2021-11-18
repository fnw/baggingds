import numpy as np
from sklearn.neighbors import NearestNeighbors

def kDNInstanceHardness(X, y,knn_classifier=None,k=5):
    num_examples = X.shape[0]

    if knn_classifier is not None:
        kNN = knn_classifier
    else:
        kNN = NearestNeighbors(k)
        kNN.fit(X)

    _, nearestNeighboursIndexes = kNN.kneighbors(n_neighbors=k)

    nearestNeighboursIndexes = nearestNeighboursIndexes.flatten()

    neighbor_classes = y[nearestNeighboursIndexes]
    neighbor_classes = neighbor_classes.reshape((num_examples, k))

    #Precisa do reshape pra forcar o broadcasting
    different_classes = neighbor_classes != y.reshape((num_examples,1))

    misclassificationLikelihood = np.sum(different_classes, axis = 1)/float(k)
    misclassificationLikelihood = misclassificationLikelihood.reshape((num_examples,1))

    return misclassificationLikelihood