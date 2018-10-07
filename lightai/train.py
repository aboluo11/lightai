from .callbacks import *
from .data import *
from .core import *

@dataclass
class Learner:
    model: nn.Module
    trn_dl: DataLoader
    optimizer: optim.Optimizer
    evaluator: Callable
    state_dir: str = 'states'
    callbacks: List[Callback] = field(default_factory=list)
    epoch: int = 0
    writer: Optional[SummaryWriter] = None
    def __post_init__(self):
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.sched: Optional[Callback] = None

    def fit(self, n_epoch: int, sched: Callback):
        self.sched = sched
        callbacks = self.callbacks + [self.sched]
        mb = master_bar(range(n_epoch))
        for epoch in mb:
            self.model.train()
            losses = []
            for x, target in progress_bar(self.trn_dl, parent=mb):
                trn_loss = self.step(x, target)
                losses.append(trn_loss)
            trn_loss = np.mean(losses)
            val_loss, metric = self.evaluator()
            param = metric if metric is not None else val_loss
            for cb in callbacks:
                cb.on_epoch_end(param)
            self.log(trn_loss, val_loss, metric, mb)
            self.epoch += 1

    def step(self, x: np.ndarray, target: np.ndarray)->float:
        predict = self.model(x)
        loss = self.loss_fn(predict, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def log(self, trn_loss: float, val_loss: float, metric: Optional[float], mb: MasterBar):
        width = 11
        precision = 6
        message = ''
        for loss in (trn_loss, val_loss):
            message += f'{loss:{width}.{precision}}'
        if metric is not None:
            message += f'{metric:{width}.{precision}}'
        mb.write(message)
        if self.writer:
            self.writer.add_scalars('loss', {
                'train': trn_loss,
                'val': val_loss
            }, self.epoch)
            if metric is not None:
                self.writer.add_scalar('metirc', metric, self.epoch)

    def save_all(self, name):
        torch.save({
            'learner': self,
        }, self.state_dir/name)