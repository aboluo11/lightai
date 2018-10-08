from ..callback import *

class PytorchScheduler(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, metric: float, **kwargs: Any):
        self.scheduler.step(metric)