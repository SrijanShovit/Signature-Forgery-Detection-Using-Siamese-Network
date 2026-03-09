import numpy as np

class ThresholdEstimator:

    def __init__(self, metrics):
        self.metrics = metrics

    def compute(self):

        thresholds = {}

        for key, values in self.metrics.items():

            values = np.array(values)

            thresholds[key] = {
                "low": float(np.percentile(values,5)),
                "high": float(np.percentile(values,95))
            }

        return thresholds