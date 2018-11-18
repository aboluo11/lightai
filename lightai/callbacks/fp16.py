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


class FP16(Callback):
    def __init__(self, model_params, master_params, loss_scale):
        self.model_params = model_params
        self.master_params = master_params
        self.loss_scale = loss_scale

    def on_train_begin(self, **kwargs):
        for model, master in zip(self.model_params, self.master_params):
            master.data.copy_(model.data)

    def on_step_begin(self, **kwargs):
        for model, master in zip(self.model_params, self.master_params):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data.copy_(model.grad.data)
            master.grad /= self.loss_scale

    def on_step_end(self, **kwargs):
        for model, master in zip(self.model_params, self.master_params):
            model.data.copy_(master.data)

    def on_backward_begin(self, loss, **kwargs):
        loss *= self.loss_scale

    def on_backward_end(self, loss, **kwargs):
        loss /= self.loss_scale


def to_fp16(learner, loss_scale):
    model = learner.model
    model.half()
    bn_to_float(model)
    model_params, master_params = get_params(model)
    learner.optimizer = learner.optim_fn(master_params)
    learner.callbacks.append(FP16(model_params, master_params, loss_scale))
