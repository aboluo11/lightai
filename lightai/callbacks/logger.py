from ..core import *
from ..callback import *


class Logger(Callback):
    def __init__(self, writer: SummaryWriter, has_metrics: bool):
        self.epoch = 0
        self.writer = writer
        self.has_metrics = has_metrics

    def on_epoch_end(self, trn_loss: float, eval_res: List[float], **kwargs):
        if self.writer:
            self.writer.add_scalars('loss', {
                'train': trn_loss,
                'val': eval_res[0]
            }, self.epoch)
            if self.has_metrics:
                self.writer.add_scalar('metirc', eval_res[1], self.epoch)
        self.epoch += 1