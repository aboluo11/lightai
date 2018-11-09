from .core import *

class DataLoaderWrapper:
    def __init__(self, dl, tsfms: List[Callable]=None):
        self.dl = dl
        if tsfms is None:
            self.tsfms = []
        else:
            self.tsfms = tsfms

    def __iter__(self):
        for batch in self.dl:
            for t in self.tsfms:
                batch = t(batch)
            yield batch

    def __len__(self):
        return len(self.dl)

    def add_tsfm(self, tsfm):
        self.tsfms.append(tsfm)

    def remove_tsfm(self, tsfm):
        self.tsfms.remove(tsfm)