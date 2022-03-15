import math

import tensorflow as tf


class OneCycleLr(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    [Implementation taken from PyTorch:
    (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR)]

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch
    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
    max_lr (float): Upper learning rate boundaries in the cycle.
    total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
    epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
    steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
    pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
    anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
    div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
    final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
    """

    def __init__(self,
                 max_lr: float,
                 total_steps: int = None,
                 epochs: int = None,
                 steps_per_epoch: int = None,
                 pct_start: float = 0.3,
                 anneal_strategy: str = "cos",
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 ) -> None:
        super().__init__()

        # validate total steps:
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError(
                "You must define either total_steps OR (epochs AND steps_per_epoch)"
            )
        if total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(
                    f"Expected non-negative integer total_steps, but got {total_steps}"
                )
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError(
                    f"Expected non-negative integer epochs, but got {epochs}"
                )
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError(
                    f"Expected non-negative integer steps_per_epoch, but got {steps_per_epoch}"
                )
            # Compute total steps
            self.total_steps = epochs * steps_per_epoch

        self.step_size_up = int(pct_start * self.total_steps) - 1
        self.step_size_down = self.total_steps - self.step_size_up - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(
                f"Expected float between 0 and 1 pct_start, but got {pct_start}"
            )

        # Validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"anneal_strategy must by one of 'cos' or 'linear', instead got {anneal_strategy}"
            )
        if anneal_strategy == "cos":
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        self.initial_lr = max_lr / div_factor
        self.max_lr = max_lr
        self.min_lr = self.initial_lr / final_div_factor

    @staticmethod
    @tf.function
    def _annealing_cos(start, end, pct) -> float:
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = tf.math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start, end, pct) -> float:
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    @tf.function
    def get_step_lr(self, step_num: int) -> float:
        """Gets the learning rate for the step the learning rate"""
        if step_num <= self.step_size_up:
            # update learning rate
            return self.anneal_func(
                self.initial_lr, self.max_lr, float(step_num) / self.step_size_up
            )

        down_step_num = step_num - self.step_size_up
        # update learning rate
        return self.anneal_func(
            self.max_lr, self.min_lr, float(down_step_num) / self.step_size_down
        )

    def __call__(self, step) -> float:
        return self.get_step_lr(step)

    def get_config(self):
        return {
            "total_steps": self.total_steps,
            "step_size_up": self.step_size_up,
            "step_size_down": self.step_size_down,
            "anneal_func": self.anneal_func,
            "initial_lr": self.initial_lr,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }
