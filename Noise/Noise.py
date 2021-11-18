import numpy as np


def label_noise(y, probability, random_state, return_indexes=False):

    numel = len(y)

    classes = set(y)
    num_classes = len(classes)

    classes = list(classes)
    classes = np.array(classes)

    new_classes = np.tile(classes, [numel, 1])

    # Verificando quais classes NAO SAO a classe atual do elemento
    # So podemos trocar a classe para uma das outras classes
    diffIndex = new_classes != np.reshape(y.T,(numel,1))
    new_classes = new_classes[diffIndex]
    new_classes = np.reshape(new_classes,(numel,num_classes-1))

    np.random.seed(random_state)

    chosen = np.random.randint(0,num_classes-1,numel)

    new_classes = new_classes[np.arange(numel), chosen]

    np.random.seed(random_state)

    # Se a probabiliade eh maior que o nivel de ruido, troque de volta para o label original
    # Eh a mesma coisa que dizer que as classes vao ser trocadas com p=ruido
    idx = np.random.rand(len(y)) > probability

    np.random.seed()

    new_classes[idx] = y[idx]

    if return_indexes:
        return new_classes, np.logical_not(idx)
    else:
        return new_classes
