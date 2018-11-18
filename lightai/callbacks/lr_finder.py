from ..core import *
from ..callback import *


class SmoothenValue:
    "Create a smooth moving average for a value (loss, etc)."

    def __init__(self, beta: float = 0.98):
        "Create smoother for value, beta should be 0<beta<1."
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def __call__(self, val: float):
        "Add current value to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        smooth = self.mov_avg / (1 - self.beta ** self.n)
        return smooth


class LRFinder(Callback):
    def __init__(self, model, optimizer, min_lr, max_lr, n_iter):
        self.smoother = SmoothenValue()
        self.optimizer = optimizer
        self.lrs = np.geomspace(min_lr, max_lr, num=n_iter, endpoint=True)
        self.losses = []
        self.iter = 0
        self.best = None
        self.save_path = Path('saved')
        self.save_path.mkdir(exist_ok=True)
        self.model = model

    def on_train_begin(self, **kwargs):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, self.save_path / 'temp')

    def on_batch_begin(self, **kwargs):
        self.optimizer.param_groups[0]['lr'] = self.lrs[self.iter]
        self.iter += 1

    def on_batch_end(self, trn_loss: float, **kwargs) -> bool:
        trn_loss = self.smoother(trn_loss)
        self.losses.append(trn_loss)
        if self.best is None or trn_loss < self.best:
            self.best = trn_loss
        if trn_loss > self.best * 2:
            return True
        return False

    def on_train_end(self, **kwargs):
        state_dict = torch.load(self.save_path / 'temp')
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def plot(self, skip_begin=0, skip_end=0):
        total_len = len(self.losses)
        plt.xscale('log')
        plt.plot(self.lrs[skip_begin:total_len - skip_end], self.losses[skip_begin:total_len - skip_end])
