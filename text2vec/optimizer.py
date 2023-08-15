
import numpy as np


class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, lr, n_warmup_steps, current_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.init_lr = lr

    def step_and_update_lr_frozen(self, learning_rate_frozen):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = learning_rate_frozen
        self._optimizer.step()

    def step_and_update_lr(self):
        # self._update_learning_rate()# don't need update lr manually
        self._optimizer.step()

    def get_learning_rate(self):
        learning_rate = 0.0
        for param_group in self._optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
