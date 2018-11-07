from .callbacks import *
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
        self.epoch = 0
        self.callbacks.append(Printer(metrics))
        self.callbacks.append(Logger(writer=self.writer, metrics=metrics))

    def fit(self, n_epoch: Optional[int]=None, sched: Optional[Callback]=None):
        callbacks = self.callbacks + [sched]
        for cb in callbacks:
            cb.on_train_begin()
        mb = master_bar(range(n_epoch))
        for epoch in mb:
            for cb in callbacks:
                cb.on_epoch_begin()
            self.model.train()
            losses = []
            for x, target in progress_bar(self.trn_dl, parent=mb):
                x, target = x.cuda(), target.cuda()
                for cb in callbacks:
                    cb.on_batch_begin()
                trn_loss = self.step(x, target)
                losses.append(trn_loss)
                for cb in callbacks:
                    stop = cb.on_batch_end(trn_loss=trn_loss)
                    if stop:
                        return
            trn_loss = np.mean(losses)
            eval_res = self.evaluator()
            self.epoch += 1
            for cb in callbacks:
                cb.on_epoch_end(trn_loss=trn_loss, eval_res=eval_res, epoch=self.epoch,
                                learner=self, bar=mb)
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
