import numpy as np

def all_classes_present(y_prime, y):
    return set(y) == set(y_prime)

def add_missing_classes(X_prime, X, y_prime, y, minimum=1, seed=None):
    missing_classes = set(y) - set(y_prime)

    for c in missing_classes:
        idx = np.where(y == c)[0]

        if seed is not None:
            np.random.seed(seed)

        idx_selected = np.random.choice(idx, size=minimum)

        X_prime = np.vstack((X_prime, X[idx_selected, :]))
        y_prime = np.concatenate((y_prime, np.array(y[idx_selected])))

    return X_prime, y_prime

def ensure_minimum_size(X, y, minimum_size, seed=None):
    classes, num_classes = np.unique(y, return_counts=True)

    X_prime, y_prime = X, y

    for c in classes:
        idx = np.where(y == c)[0]

        num_elements = len(idx)

        if num_elements < minimum_size:
            missing = minimum_size - num_elements

            if seed is not None:
                np.random.seed(seed)

            idx_selected = np.random.choice(idx, size=missing)

            X_prime = np.vstack((X_prime, X[idx_selected, :]))
            y_prime = np.concatenate((y_prime, np.array(y[idx_selected])))

    return X_prime, y_prime
