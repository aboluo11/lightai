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
        mb = master_bar(range(n_epoch))
        for cb in callbacks:
            cb.on_train_begin(mb=mb)
        for epoch in mb:
            self.model.train()
            for cb in callbacks:
                cb.on_epoch_begin()
            losses = []
            for x, target in progress_bar(self.trn_dl, parent=mb):
                x, target = x.cuda(), target.cuda()
                for cb in callbacks:
                    cb.on_batch_begin(x=x, target=target)
                trn_loss = self.step(x, target, callbacks)
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
                                learner=self, mb=mb)
        for cb in callbacks:
            cb.on_train_end()
        self.sched = None

    def step(self, x: np.ndarray, target: np.ndarray, callbacks)->float:
        predict = self.model(x)
        predict = predict.float()
        loss = self.loss_fn(predict, target)
        self.optimizer.zero_grad()
        for cb in callbacks:
            cb.on_backward_begin(loss=loss)
        loss.backward()
        self.optimizer.step()
        return loss.item()
