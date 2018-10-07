from ..callback import *

class ReduceOnPlateau(Callback):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold,
                                                       threshold_mode, cooldown, min_lr, eps)

    def on_epoch_end(self, metric: float, **kwargs: Any):
        self.scheduler.step(metric)