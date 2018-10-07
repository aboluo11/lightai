from ..core import *
from ..callback import *


class SaveBestModel(Callback):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, small_better: bool, model_dir: str='models',
                 name: str='best'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.path = self.model_dir/name
        self.small_better = small_better
        self.best_metric = None
        self.model = model
        self.optimizer = optimizer

    def on_epoch_end(self, metric: float, **kwargs: Any):
        if self.small_better:
            metric = -metric
        if not self.best_metric or metric >= self.best_metric:
            self.best_metric = metric
            self.learner.save_model(self.name)
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, self.path)