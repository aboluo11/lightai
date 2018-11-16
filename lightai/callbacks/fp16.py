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


class HalfInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.half()


class OptimWrapper:
    def __init__(self, optimizer, model_params, master_params, loss_scale):
        self.model_params = model_params
        self.master_params = master_params
        self.loss_scale = loss_scale
        self.optimizer = optimizer.__class__(master_params)

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


def to_fp16(learner, loss_scale):
    model = learner.model
    model.half()
    bn_to_float(model)
    model_params, master_params = get_params(model)
    learner.model = nn.Sequential(HalfInput(), model)
    learner.optimizer = OptimWrapper(learner.optimizer, model_params, master_params, loss_scale)
