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
    has_metrics: bool
    state_dir: str = 'states'
    callbacks: List[Callback] = field(default_factory=list)
    writer: Optional[SummaryWriter] = None
    def __post_init__(self):
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.sched: Optional[Callback] = None
        self.callbacks.append(Printer(self.has_metrics))
        self.callbacks.append(Logger(writer=self.writer, has_metrics=self.has_metrics))

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
            eval_res = self.evaluator()
            for cb in callbacks:
                cb.on_epoch_end(trn_loss=trn_loss, eval_res=eval_res, elapsed_time=time.time()-start, epoch=epoch)
        for cb in callbacks:
            cb.on_train_end()

    def step(self, x: np.ndarray, target: np.ndarray)->float:
        predict = self.model(x)
        loss = self.loss_fn(predict, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_all(self, name):
        torch.save({
            'learner': self,
        }, self.state_dir/name)