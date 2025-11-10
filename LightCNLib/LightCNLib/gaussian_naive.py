import numpy
try:
    import cupy
except:
    pass
if cupy is not None and cupy.is_available():
    np = cupy
else:
    np = numpy

import math

class GaussNaiveBayes:
    def __init__(self, classes):
        self.classes = classes
        self.class_counts = np.zeros((len(classes)))
        self.sums = None
        self.sumsq = None
    def p_fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.sums is None:
            self.sums = np.zeros((len(self.classes), X.shape[1]))
            self.sumsq = np.zeros((len(self.classes), X.shape[1]))
        for i, c in enumerate(self.classes):
            idxs = np.where(y == c)[0]
            self.class_counts[i] += len(idxs)
            self.sums[i] += np.sum(X[idxs], axis=0)
            self.sumsq[i] = np.sum(X[idxs] ** 2, axis=0)
    
    def finish(self):
        t = self.class_counts.sum()
        self.prior = self.class_counts / t
        self.mean = self.sums / self.class_counts[:, None]
        self.vari = (self.sumsq / self.class_counts[:, None]) - self.mean ** 2    
    
    def predict(self, X):
        X = np.asarray(X)
        log_pr = np.log(self.prior)
        t1 = -0.5 * np.log(2 * np.pi * self.vari)
        y_pred = np.zeros((X.shape[0]))
        for n in range(X.shape[0]):
            d = (X[n] - self.mean) ** 2 / self.vari
            log_l = np.sum(t1 - d * 0.5, axis=1)
            log_po = log_pr + log_l
            y_pred[n] = self.classes[np.argmax(log_po)]
        return y_pred

