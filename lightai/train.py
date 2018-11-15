from .callbacks import *
from .core import *

class Learner:
    def __init__(self, model: nn.Module, trn_dl: DataLoader, val_dl: DataLoader,
                 optimizer: optim.Optimizer, loss_fn: Callable, metrics: List,
                 callbacks: List[Callback]=[], writer: Optional[SummaryWriter]=None):
        self.model = model
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.writer = writer
        self.epoch = 0
        self.metrics = metrics
        self.callbacks.append(Printer(metrics))
        self.callbacks.append(Logger(writer=self.writer, metrics=metrics))

    def fit(self, n_epoch: Optional[int]=None, sched: Optional[Callback]=None, loss_scale=512):
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
                trn_loss = self.step(x, target, loss_scale)
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
        self.sched = None

    def step(self, x: np.ndarray, target: np.ndarray, loss_scale)->float:
        predict = self.model(x)
        predict = predict.float()
        loss = self.loss_fn(predict, target)*loss_scale
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()/loss_scale/target.shape[0]

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