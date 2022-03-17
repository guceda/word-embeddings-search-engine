import pandas as pd

class TSVDocumentReader(object):
    def __init__(self, path):
        self.path = path

    @property
    def corpus(self):
        df = pd.read_csv(self.path, delimiter="\t", header=None)
        return df[3].values.tolist()
