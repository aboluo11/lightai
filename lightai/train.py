from .callbacks import *
from .data import *
from .core import *

@dataclass
class Learner:
    model: nn.Module
    trn_dl: DataLoader
    optimizer: optim.Optimizer
    evaluator: Callable
    loss_fn: Callable
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
        for epoch in range(n_epoch):
            start = time.time()
            for cb in callbacks:
                cb.on_epoch_begin()
            self.model.train()
            losses = []
            for x, target in self.trn_dl:
                trn_loss = self.step(x, target)
                losses.append(trn_loss)
            trn_loss = np.mean(losses)
            val_loss, metrics = self.evaluator()
            param = metrics if metrics is not None else val_loss
            for cb in callbacks:
                cb.on_epoch_end(metrics=param)
            self.log(trn_loss, val_loss, metrics, start, epoch+1)
            self.epoch += 1

    def step(self, x: np.ndarray, target: np.ndarray)->float:
        predict = self.model(x)
        loss = self.loss_fn(predict, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def log(self, trn_loss: float, val_loss: float, metrics: Optional[float], start: float, epoch: int):
        width = 11
        precision = 6
        message = ''
        for loss in (trn_loss, val_loss):
            message += f'{loss:{width}.{precision}}'
        if metrics is not None:
            message += f'{metrics:{width}.{precision}}'
        elapsed_time = time.time() - start
        message += f'{elapsed_time:{width}.{precision}}'
        print(message)
        if self.writer:
            self.writer.add_scalars('loss', {
                'train': trn_loss,
                'val': val_loss
            }, self.epoch)
            if metrics is not None:
                self.writer.add_scalar('metirc', metrics, self.epoch)

    def save_all(self, name):
        torch.save({
            'learner': self,
        }, self.state_dir/name)