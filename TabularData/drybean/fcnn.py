import numpy as np
import matplotlib.pyplot as plt

class FCNN:
    def __init__(self):
        # store the selected subset and its labels. These will be populated during the fit method
        self.subset = None
        self.subset_labels = None

    # calculates the Euclidean distance between two points
    def _dist(self, x, y):
        return np.linalg.norm(x - y)

    # fits the FCNN model to the training data
    def fit(self, train, label, alpha = 0.95):
        train = np.asarray(train)  # shape (n, d)
        label = np.asarray(label).flatten()  # shape (n,)
        n, d = train.shape

        unique_labels = np.unique(label)
        num_label = unique_labels.size
        label_map = {v: i for i, v in enumerate(unique_labels)} # creates a dictionary that maps each unique label to a sequential index. This mapping ensures that our labels are consecutive integers starting from 0
        mapped_label = np.vectorize(label_map.get)(label) # maps the original labels to their corresponding mapped indices


        # Calculates the centroid (mean point) for each class
        # For each class, sums up all points and divides by count
        center = np.zeros((num_label, d))
        count = np.zeros(num_label, dtype=int)
        class_counts = np.zeros(num_label)

        for i in range(n):
            c = mapped_label[i]

            class_counts[c] += 1

            center[c] += train[i]
            count[c] += 1

        for c in range(num_label):
            if count[c] > 0:
                center[c] /= count[c]



        # Find medians
        # For each class, finds the point that is closest to the centroid
        median = np.full(num_label, -1, dtype=int)
        dist_median = np.full(num_label, np.inf)

        for i in range(n):
            c = mapped_label[i]
            dst = self._dist(train[i], center[c])
            if dst < dist_median[c]:
                dist_median[c] = dst
                median[c] = i



        # Initializes the subset with the median points from each class
        subset = []
        for c in range(num_label):
            if count[c] > 0:
                subset.append(median[c])


        # Removes duplicates from the subset
        subset = list(set(subset))

        # Initializes the first and last indices of the subset
        delta_first = 0
        delta_last = len(subset)

        # Initializes arrays to store the nearest neighbor and its distance for each point
        nearest = np.full(n, -1)
        dist_nearest = np.full(n, np.inf)

        # Initializes arrays to store the representative point and its distance for each subset point
        rep = np.full(n, -1)
        dist_rep = np.full(n, -1.0)

        visited = np.full(n, False)

        # Initializes the error counter to the total number of points
        error = n

        # Calculate class-wise accuracy
        class_errors = np.zeros(num_label)

        # Loops until all classes have error rate less than (1-alpha)
        while True:

            error = 0 # error of the current iteration
            class_errors = np.zeros(num_label)
            
            rep = np.full(len(subset), -1)
            dist_rep = np.full(len(subset), -1.0)

            for i in range(n):
                if not visited[i]:
                    for c in range(delta_first, delta_last):
                        idx = subset[c]
                        dst = self._dist(train[i], train[idx])
                        if dst < dist_nearest[i]:
                            dist_nearest[i] = dst
                            nearest[i] = c


                    c = nearest[i]
                    if mapped_label[i] != mapped_label[subset[c]]:

                        error += 1
                        class_errors[mapped_label[i]] += 1

                        if rep[c] == -1 or dist_nearest[i] < dist_rep[c]:
                            rep[c] = i
                            dist_rep[c] = dist_nearest[i]


            # Calculate error rates for each class
            class_error_rates = class_errors / class_counts
            
            # Check if any class has error rate greater than (1-alpha)
            max_error_rate = np.max(class_error_rates)
            if max_error_rate <= alpha:
                break


            delta_first = delta_last
            for c in range(delta_first):
                if rep[c] != -1:
                    subset.append(rep[c])
                    visited[rep[c]] = True
                    delta_last += 1

            print(f"subset size = {delta_first}, errors = {error}, accuracy = {100*(1-error/n):.2f}")

            class_accuracy = 1 - class_error_rates
            for c in range(num_label):
                print(f"Class {unique_labels[c]}: {100*class_accuracy[c]:.2f}%")


        reduced_ratio = 100*(n-len(subset))/n
        print(f"Reduced ratio: {reduced_ratio:.2f}%")
        
        # Extract final subset
        set_data = train[subset]
        set_label = label[subset]

        self.subset = set_data
        self.subset_labels = set_label
        return set_data, set_label, reduced_ratio
    

"""
Gli errori per classe possono fluttuare perché:
- Quando aggiungiamo un nuovo punto al subset, questo può influenzare la classificazione di punti di diverse classi
- Il punto più vicino (nearest neighbor) per un punto può cambiare quando aggiungiamo nuovi punti al subset
- Dovremmo mantenere traccia degli errori cumulativi per classe tra le iterazioni
"""
