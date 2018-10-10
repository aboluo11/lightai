from .callbacks import *
from .data import *
from .core import *

class Learner:
    def __init__(self, model: nn.Module, trn_dl: DataLoader, optimizer: optim.Optimizer,
                 evaluator: Callable, loss_fn: Callable, metrics: List,
                 callbacks: List[Callback]=[], writer: Optional[SummaryWriter]=None):
        self.model = model
        self.trn_dl = trn_dl
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.writer = writer
        self.sched: Optional[Callback] = None
        self.callbacks.append(Printer(metrics))
        self.callbacks.append(Logger(writer=self.writer, metrics=metrics))

    def fit(self, n_epoch: int, sched: Optional[Callback]=None):
        assert self.sched or sched
        if sched:
            self.sched = sched
        callbacks = self.callbacks + [self.sched]
        for cb in callbacks:
            cb.on_train_begin()
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
                cb.on_epoch_end(trn_loss=trn_loss, eval_res=eval_res, elapsed_time=time.time()-start, epoch=epoch,
                                learner=self)
        for cb in callbacks:
            cb.on_train_end()
        self.sched = None

    def step(self, x: np.ndarray, target: np.ndarray)->float:
        predict = self.model(x)
        loss = self.loss_fn(predict, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
