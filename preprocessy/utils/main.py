def num_of_samples(X):
    if hasattr(X, "__len__"):
        return len(X)
