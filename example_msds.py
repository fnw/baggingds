import time

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier

from InstanceHardness.InstanceHardness import kDNInstanceHardness
from BaggingSelectionProbabilities import linearProbabilitiesFromHardness

from Noise.Noise import label_noise
from DynamicSelection.DynamicSelection import create_selector2

from DynamicSelection.BaggingDS import BaggingDS


def main():
    # Making data

    X, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, n_repeated=0)
    percentage_train = 0.75

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - percentage_train,
                                                        test_size=percentage_train, stratify=y)

    # Adding noise
    y_train = label_noise(y_train, 0.2, random_state=42)

    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=1 - percentage_train,
                                                      test_size=percentage_train, stratify=y_train)

    # Calculating instance hardness
    hardness = kDNInstanceHardness(X_val, y_val)
    probabilities = linearProbabilitiesFromHardness(hardness)

    base_pool = BaggingClassifier(base_estimator=Perceptron(max_iter=100, random_state=123987), n_estimators=50,
                                  random_state=123987)

    # Static Selection
    print('Fitting base pool...')
    base_pool.fit(X_train, y_train)

    print('Base pool acc: ', accuracy_score(y_test, base_pool.predict(X_test)))

    # OLA
    base_selector = 'OLA'

    ola_predictor = create_selector2(base_selector, base_pool)

    ola_predictor.fit(X_val, y_val)

    print('OLA acc: ', accuracy_score(y_test, ola_predictor.predict(X_test)))

    # BagDS
    selector = BaggingDS(base_pool, base_selector, n_bags=50, probabilities=None, jobs=4)

    print('Fitting BagDS...')

    beg = time.time()
    selector.fit(X_val, y_val)

    print('Predicting...')
    y_pred = selector.predict(X_test)

    print('Time elapsed: ', time.time() - beg)

    accuracy = accuracy_score(y_test, y_pred)

    print('Accuracy new: ', accuracy)
    print('\n')

    # MSDS
    selector = BaggingDS(base_pool, base_selector, n_bags=50, probabilities=probabilities, jobs=4)

    print('Fitting MSDS...')

    beg = time.time()
    selector.fit(X_val, y_val)

    print('Predicting...')
    y_pred = selector.predict(X_test)

    print('Time elapsed: ', time.time() - beg)

    accuracy = accuracy_score(y_test, y_pred)

    print('Accuracy new: ', accuracy)
    print('\n')


if __name__ == '__main__':
    main()
