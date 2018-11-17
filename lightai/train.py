from .callbacks import *
from .core import *


class Learner:
    def __init__(self, model: nn.Module, trn_dl: DataLoader, val_dl: DataLoader,
                 optim_fn: optim.Optimizer, loss_fn: Callable, metrics: List,
                 callbacks: List[Callback] = [], writer: Optional[SummaryWriter] = None):
        self.model = model
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.optim_fn = optim_fn
        self.optimizer = optim_fn(model.parameters())
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.writer = writer
        self.epoch = 0
        self.metrics = metrics
        self.callbacks.append(Printer(metrics))
        self.callbacks.append(Logger(writer=self.writer, metrics=metrics))

    def fit(self, n_epoch: Optional[int] = None, sched: Optional[Callback] = None):
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
                trn_loss = self.step(x, target)
                losses.append(trn_loss)
                for cb in callbacks:
                    stop = cb.on_batch_end(trn_loss=trn_loss)
                    if stop:
                        return
            trn_loss = np.mean(losses)
            eval_res = self.evaluate()
            self.epoch += 1
            for cb in callbacks:
                cb.on_epoch_end(trn_loss=trn_loss, eval_res=eval_res, epoch=self.epoch,
                                learner=self, mb=mb)
        for cb in callbacks:
            cb.on_train_end()

    def step(self, x: np.ndarray, target: np.ndarray) -> float:
        predict = self.model(x)
        predict = predict.float()
        true_loss = self.loss_fn(predict, target)
        self.optimizer.zero_grad()
        for cb in self.callbacks:
            a = cb.ob_backward_begin(true_loss)
            if a is not None:
                loss = a
        loss.backward()
        for cb in self.callbacks:
            a = cb.ob_backward_end(loss)
            if a is not None:
                true_loss = a
        for cb in self.callbacks:
            cb.on_step_begin()
        self.optimizer.step()
        for cb in self.callbacks:
            cb.on_step_end()
        return true_loss.item()

    def evaluate(self):
        self.model.eval()
        losses = []
        bses = []
        with torch.no_grad():
            for x, target in self.val_dl:
                x, target = x.cuda(), target.cuda()
                predict = self.model(x)
                predict = predict.float()
                for metric in self.metrics:
                    metric(predict, target)
                losses.append(self.loss_fn(predict, target))
                bses.append(target.shape[0])
            loss = np.average(torch.stack(losses).cpu().numpy(), weights=bses)
            res = [loss] + [metric.res() for metric in self.metrics]
            return res
