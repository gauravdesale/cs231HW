import numpy as np
import math as math
class KNearestNeighbors(object):
    Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
    Yval = Ytr[:1000]
    Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
    Ytr = Ytr[1000:]
def __init__(self):
        pass
def train(self, X):
        shapeHorizontal = X.shape[0]
        predictedLabels = np.zeros(shapeHorizontal)

def compute_distance_one_loop():
    for x in range(shapeHorizontal):
            distances = math.sqrt(math.sum(math.abs((self.Xrt - x[i,:], axis=1)**2)))
            min_index = np.argmin(distances)
            dists = self.Ytr[min_index]
        return dists
def predict_labels(dists, k):
    clf.fit(Xval_rows, Yval)
    #this one is kinda ghetto still
def sum(dists):





