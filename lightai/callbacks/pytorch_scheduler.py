from ..callback import *


class ReduceOnPlateau(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, eval_res, **kwargs):
        self.scheduler.step(eval_res[-1])


class LRSchedWrapper(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_begin(self, **kwargs):
        self.scheduler.step()
