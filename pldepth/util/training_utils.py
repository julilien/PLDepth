class LearningRateScheduleProvider(object):
    def __init__(self, steps=None, init_lr=1e-3, multiplier=0.1, warmup=0):
        if steps is None:
            self.steps = [80, 120, 160, 180]
        else:
            self.steps = steps
        self.init_lr = init_lr
        self.multiplier = multiplier
        self.warmup = warmup

    def get_lr_schedule(self, epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        if self.warmup > 0 and epoch < self.warmup:
            return (epoch + 1) * self.init_lr / self.warmup

        lr = self.init_lr
        multiplier = self.multiplier
        for loc_steps in self.steps:
            if epoch >= loc_steps:
                lr *= multiplier
            else:
                break

        return lr
