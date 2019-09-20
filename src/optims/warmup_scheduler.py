from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, lr:float, num_warmup_iters:int, warmup_factor:float):
        self.lr = lr
        self.optimizer = optimizer
        self.num_warmup_iters = num_warmup_iters
        self.warmup_factor = warmup_factor

        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        self._check_state()

        # Linear interpolation from [self.warmup_factor * self.lr] to [self.lr]
        alpha = self.last_epoch / self.num_warmup_iters
        lr = self.lr * (self.warmup_factor * (1 - alpha) + alpha)

        return [lr] * len(self.base_lrs)

    def _check_state(self):
        assert self.last_epoch >= 0
        assert self.last_epoch <= self.num_warmup_iters, \
            f'You are trying to use warmup after warmup period:' \
            f'Last epoch: {self.last_epoch}. Num warmup iters: {self.num_warmup_iters}'
