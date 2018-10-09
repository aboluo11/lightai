from ..core import *
from ..callback import *


class SavePeriodically(Callback):
    def __init__(self, period: int, state_dir: str='states', name: str='saved'):
        self.period = period
        state_dir = Path(state_dir)
        state_dir.mkdir(exist_ok=True)
        self.path = state_dir/name

    def on_epoch_end(self, epoch, learner, **kwargs):
        if epoch % self.period == 0:
            torch.save(learner, self.path, dill)