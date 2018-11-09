from ..core import *
from ..callback import *

def get_params(model):
    model_params = [param for param in model.parameters() if param.requires_grad]
    master_params = [param.clone().float().detach() for param in model_params]
    return model_params, master_params

def bn_to_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        bn_to_float(child)

def batch_to_half(batch):
    x, target = batch
    return [x.half(), target]

class tofp16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.half()

class OptimWrapper:
    def __init__(self, optimizer, model_params, master_params, loss_scale):
        self.model_params = model_params
        self.master_params = master_params
        self.loss_scale = loss_scale
        self.optimizer = optimizer

    def step(self):
        for model, master in zip(self.model_params, self.master_params):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data.copy_(model.grad.data)
            master.grad /= self.loss_scale
        self.optimizer.step()
        for model, master in zip(self.model_params, self.master_params):
            model.data.copy_(master.data)

    def zero_grad(self):
        self.optimizer.zero_grad()

class FP16(Callback):
    def __init__(self, learner, loss_scale):
        self.learner = learner
        self.loss_scale = loss_scale

    def on_train_begin(self, **kwargs):
        model = self.learner.model
        model_params, master_params = get_params(model)
        model.half()
        bn_to_float(model)
        self.learner.model = nn.Sequential(tofp16(), model)
        self.learner.optimizer = OptimWrapper(self.learner.optimizer, model_params, master_params, self.loss_scale)

    def on_loss_begin(self, predict, **kwargs):
        predict.float()

    def on_backward_begin(self, loss, **kwargs):
        loss *= self.loss_scale

    def on_train_end(self, **kwargs):
        self.learner.trn_dl.remove_tsfm(batch_to_half)
        self.learner.evaluator.val_dl.remove_tsfm(batch_to_half)


def to_fp16(learner, loss_scale):
    fp16 = FP16(learner, loss_scale)
    learner.callbacks.append(fp16)
