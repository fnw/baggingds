from multiprocessing import Pool

import numpy as np
from scipy.stats import mode

from Util.SafeKFold import all_classes_present, add_missing_classes, ensure_minimum_size
from DynamicSelection.DynamicSelection import create_selector2

class BaggingDS:
    def __init__(self, pool_classifiers, selector, n_bags=50, k=7, DFP=False, knn=None, probabilities=None, jobs=None):
        self.pool_classifiers = pool_classifiers
        self.selector = selector
        self.n_bags = n_bags
        self.k = k
        self.DFP = DFP
        self.knn = knn
        self.probabilities = probabilities
        self.selectors = []
        self.jobs = jobs
        self.idx = None
        self.predictions = None

    def _calculate_similarities(self):
        self.idx = np.array(self.idx)

        # Making a boolean array of which elements were selected
        selected = np.zeros_like(self.idx, dtype=bool)

        for i in range(selected.shape[0]):
            selected[i, self.idx[i, :]] = True

        # Getting the unique elements for each bag and how many times they show up
        values = []
        counts = []

        for i in range(self.idx.shape[0]):
            v, c = np.unique(self.idx[i, :], return_counts=True)
            values.append(v)
            counts.append(c)

        # Making an array that shows how many times each element was selected on each bag
        all_values = np.zeros_like(self.idx)

        for i in range(self.idx.shape[0]):
            num_unique = values[i].shape[0]

            for j in range(num_unique):
                idx = values[i][j]
                all_values[i, idx] = counts[i][j]

        average_similarity = np.zeros(shape=self.idx.shape[0])

        # Here we determine which elements from a given bag are also present in any other bag.
        for i in range(selected.shape[0]):
            intersection = np.logical_and(selected[i, :], np.delete(selected, i, axis=0))

            similarity = 0

            for j in range(intersection.shape[0]):
                sum = np.sum(all_values[i, intersection[j, :]])
                sum /= self.idx.shape[1]
                similarity += sum

            similarity /= intersection.shape[0]

            average_similarity[i] = similarity

        return np.mean(average_similarity)

    def _fit_single(self, X, y, calculate_similarities=False):
        selected_idx = []
        for _ in range(self.n_bags):
            # bootstrap
            if self.probabilities is not None:
                idx = np.random.choice(X.shape[0], X.shape[0], replace=True, p=self.probabilities.reshape(self.probabilities.shape[0]))
            else:
                idx = np.random.choice(X.shape[0], X.shape[0], replace=True, p=None)

            selected_idx.append(idx)

            data, target = X[idx, :], y[idx]

            if not all_classes_present(target, y):
                data, target = add_missing_classes(data, X, target, y)

            if self.knn == 'knne':
                num_classes = len(np.unique(target))
                minimum_per_class = int(self.k/num_classes) + 1
                data, target = ensure_minimum_size(data, target, minimum_per_class)

            new_selector = create_selector2(self.selector, self.pool_classifiers, k=self.k, DFP=self.DFP, knn=self.knn)

            new_selector.fit(data, target)

            self.selectors.append(new_selector)

        if calculate_similarities:
            self.idx = selected_idx
            return self._calculate_similarities()

    def _fit_bag(self, X, y):
        new_selector = create_selector2(self.selector, self.pool_classifiers, k=self.k, DFP=self.DFP, knn=self.knn)

        new_selector.fit(X, y)

        return new_selector

    def _generate_bag(self, X, y, return_idx=False):
        if self.probabilities is not None:
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True,
                               p=self.probabilities.reshape(self.probabilities.shape[0]))
        else:
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True, p=None)

        data, target = X[idx, :], y[idx]

        if not all_classes_present(target, y):
            data, target = add_missing_classes(data, X, target, y)

        if self.knn == 'knne':
            num_classes = len(np.unique(target))
            minimum_per_class = int(self.k / num_classes) + 1
            data, target = ensure_minimum_size(data, target, minimum_per_class)

        if return_idx:
            return data, target, idx
        else:
            return data, target

    def fit(self, X, y, calculate_similarities=False):
        if self.jobs is not None:
            # print('Jobs: ', self.jobs)
            p = Pool(self.jobs)

            if calculate_similarities:
                ret = [self._generate_bag(X, y, return_idx=True) for _ in range(self.n_bags)]

                bags = [(e[0], e[1]) for e in ret]
                idx = [e[2] for e in ret]

                self.idx = idx

                selectors = p.starmap(self._fit_bag, bags)
                self.selectors = selectors

                similarities = self._calculate_similarities()
                return similarities
            else:
                generator = (self._generate_bag(X, y) for _ in range(self.n_bags))
                selectors = p.starmap(self._fit_bag, generator)

            self.selectors = selectors

            p.close()
        else:
            if calculate_similarities:
                return self._fit_single(X, y, calculate_similarities=True)
            else:
                self._fit_single(X, y, calculate_similarities=False)

    def _predict_one_classifier(self, clf, X):
        return clf.predict(X)

    def _predict_proba_one_classifier(self, clf, X):
        return clf.predict_proba(X)

    def _predict_single(self, X):
        predictions = []

        for i in range(self.n_bags):
            predictions.append(self.selectors[i].predict(X))

        predictions = np.array(predictions)

        return predictions

    def _predict_proba_single(self, X):
        probabilities = []

        for i in range(self.n_bags):
            probabilities.append(self.selectors[i].predict_proba(X))

        probabilities = np.array(probabilities)

        probabilities = np.mean(probabilities, axis=0)

        return probabilities

    def predict_proba(self, X):
        if self.jobs is not None:
            # print('Predict jobs: ', self.jobs)
            p = Pool(self.jobs)

            generator = ((self.selectors[i], X) for i in range(self.n_bags))
            probabilities = p.starmap(self._predict_proba_one_classifier, generator)

            probabilities = np.array(probabilities)

            probabilities = np.mean(probabilities, axis=0)

            p.close()
        else:
            probabilities = self._predict_proba_single(X)

        return probabilities

    def predict(self, X, n_bags=None):
        if self.predictions is not None:
            predictions = self.predictions

        else:
            if self.jobs is not None:
                # print('Predict jobs: ', self.jobs)
                p = Pool(self.jobs)

                generator = ((self.selectors[i], X) for i in range(self.n_bags))
                predictions = p.starmap(self._predict_one_classifier, generator)

                predictions = np.array(predictions)

                p.close()
            else:
                predictions = self._predict_single(X)

        if self.predictions is None:
            self.predictions = predictions

        if n_bags is not None:
            if n_bags != self.n_bags:
                idx_predictions = np.random.choice(self.n_bags, size=n_bags, replace=False)
                predictions = predictions[idx_predictions, :]

        predictions = mode(predictions)[0]
        predictions = predictions[0, :]

        return predictions


class BaggingDSMode:
    def __init__(self, pool_classifiers, selector, probabilities, n_bags=50, k=7):
        self.pool_classifiers = pool_classifiers
        self.selector = selector
        self.n_bags = n_bags
        self.k = k
        self.probabilities = probabilities
        self.single_selector = None
        self.idx = None

    def _calculate_similarities(self):
        self.idx = np.array(self.idx)

        # Making a boolean array of which elements were selected
        selected = np.zeros_like(self.idx, dtype=bool)

        for i in range(selected.shape[0]):
            selected[i, self.idx[i, :]] = True

        # Getting the unique elements for each bag and how many times they show up
        values = []
        counts = []

        for i in range(self.idx.shape[0]):
            v, c = np.unique(self.idx[i, :], return_counts=True)
            values.append(v)
            counts.append(c)

        # Making an array that shows how many times each element was selected on each bag
        all_values = np.zeros_like(self.idx)

        for i in range(self.idx.shape[0]):
            num_unique = values[i].shape[0]

            for j in range(num_unique):
                idx = values[i][j]
                all_values[i, idx] = counts[i][j]

        average_similarity = np.zeros(shape=self.idx.shape[0])

        # Here we determine which elements from a given bag are also present in any other bag.
        for i in range(selected.shape[0]):
            intersection = np.logical_and(selected[i, :], np.delete(selected, i, axis=0))

            similarity = 0

            for j in range(intersection.shape[0]):
                sum = np.sum(all_values[i, intersection[j, :]])
                sum /= self.idx.shape[1]
                similarity += sum

            similarity /= intersection.shape[0]

            average_similarity[i] = similarity

        return np.mean(average_similarity)

    def _fit_single(self, X, y, calculate_similarities=False):
        selected_idx = []

        for _ in range(self.n_bags):
            # bootstrap
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True, p=None)

            selected_idx.append(idx)

        counts = [0 for _ in range(X.shape[0])]

        array_idx = np.array(selected_idx)

        f = 1.0/X.shape[0]

        for elem in array_idx.flat:
            counts[elem] += f

        counts = [c/self.n_bags for c in counts]
        counts = np.array(counts)

        sorted_counts_idx = np.argsort(counts)[::-1]
        counts = counts[sorted_counts_idx]

        idx = np.empty(X.shape[0], dtype=int)

        total_selected = 0
        i = 0

        while total_selected < X.shape[0]:
            this_count = int(np.round(counts[i] * X.shape[0]))

            if this_count < 1:
                this_count = 1

            if total_selected + this_count > X.shape[0]:
                this_count = X.shape[0] - total_selected

            idx[total_selected:total_selected + this_count] = sorted_counts_idx[i]

            total_selected += this_count
            i += 1

        data, target = X[idx, :], y[idx]

        if not all_classes_present(target, y):
            data, target = add_missing_classes(data, X, target, y)

        new_selector = create_selector(self.selector, self.pool_classifiers, data, target, k=self.k)

        new_selector.fit(data, target)

        self.single_selector = new_selector

        if calculate_similarities:
            self.idx = selected_idx
            return self._calculate_similarities()

    def fit(self, X, y, calculate_similarities=False):
        if calculate_similarities:
            return self._fit_single(X, y, calculate_similarities=True)
        else:
            self._fit_single(X, y, calculate_similarities=False)

    def _predict_proba_one_classifier(self, clf, X):
        return clf.predict_proba(X)

    def _predict_proba_single(self, X):
        probabilities = []

        for i in range(self.n_bags):
            probabilities.append(self.selectors[i].predict_proba(X))

        probabilities = np.array(probabilities)

        probabilities = np.mean(probabilities, axis=0)

        return probabilities

    def predict_proba(self, X):
        if self.jobs is not None:
            print('Predict jobs: ', self.jobs)
            p = Pool(self.jobs)

            generator = ((self.selectors[i], X) for i in range(self.n_bags))
            probabilities = p.starmap(self._predict_proba_one_classifier, generator)

            probabilities = np.array(probabilities)

            probabilities = np.mean(probabilities, axis=0)

            p.close()
        else:
            probabilities = self._predict_proba_single(X)

        return probabilities

    def predict(self, X):
        return self.single_selector.predict(X)



