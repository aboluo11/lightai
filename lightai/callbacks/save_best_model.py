from ..core import *
from ..callback import *


class SaveBestModel(Callback):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, small_better: bool, model_dir: str='models',
                 name: str='best'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.path = self.model_dir/name
        self.small_better = small_better
        self.best_metrics = None
        self.model = model
        self.optimizer = optimizer

    def on_epoch_end(self, metrics: float, **kwargs):
        if self.small_better:
            metrics = -metrics
        if not self.best_metrics or metrics >= self.best_metrics:
            self.best_metrics = metrics
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, self.path)