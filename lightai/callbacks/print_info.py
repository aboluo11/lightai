from ..core import *
from ..callback import *


class Printer(Callback):
    def __init__(self, has_metrics):
        self.layout = '{:^11}' + '{:11.6f}' * (4 if has_metrics else 3)

    def on_epoch_end(self, trn_loss: float, eval_res: List[float], elapsed_time: float, epoch: int, **kwargs):
        print(self.layout.format(epoch, trn_loss, *eval_res, elapsed_time))