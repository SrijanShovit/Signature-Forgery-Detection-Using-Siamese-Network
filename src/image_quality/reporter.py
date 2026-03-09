import pandas as pd

class QualityReporter:

    def __init__(self, records):
        self.df = pd.DataFrame(records)

    def global_summary(self):

        return self.df.describe()

    def classwise_summary(self):

        if "label" not in self.df.columns:
            return None

        return self.df.groupby("label").mean()

    def image_table(self):

        return self.df