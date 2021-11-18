import numpy as np

def normalizeProbabilities(initialProbabilities):
    return initialProbabilities/np.sum(initialProbabilities)

def linearProbabilitiesFromHardness(hardness):
    numElements = hardness.shape[0]

    initialProbability = 1.0/numElements * np.ones([numElements,1],dtype=float)
    initialProbability = initialProbability + (1 - hardness)

    return normalizeProbabilities(initialProbability)

def softmaxProbabilitiesFromHardness(hardness):
    e = np.exp(1 - hardness)

    return e/np.sum(e)