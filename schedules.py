import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

from keras.src.backend.torch.core import *


# Learning rate schedule with warmup, decay, then constant final lr
class LinearWarmup(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        starting_lr=0.001,
        warmup_lr=0.1,
        final_lr=0.0001,
        warmup_steps=2000,
        decay_steps=10000,
    ):
        # Store the main learning rate values
        self.starting_lr = starting_lr
        self.warmup_lr = warmup_lr
        self.final_lr = final_lr

        # Store the number of steps for warmup and decay
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        # Keep track of the latest training step
        self.last_step = 0
        self.name = "LinearWarmup"

        # First line equation for increasing the learning rate during warmup
        self.a1 = (self.warmup_lr - self.starting_lr) / self.warmup_steps
        self.b1 = self.starting_lr

        # Second line equation for decreasing the learning rate during decay
        self.a2 = (self.final_lr - self.warmup_lr) / self.decay_steps
        self.b2 = self.final_lr - self.a2 * (self.decay_steps + self.warmup_steps)

    def get_lr(self, step):
        # Increase learning rate during the warmup stage
        if step < self.warmup_steps:
            return self.a1 * step + self.b1

        # Decrease learning rate during the decay stage
        elif step <= self.warmup_steps + self.decay_steps:
            return self.a2 * step + self.b2

        # Keep the learning rate constant after warmup and decay
        else:
            return self.final_lr

    def __call__(self, step):
        # Update the latest step and return the learning rate
        self.last_step = step
        return self.get_lr(step)

    def get_config(self):
        # Return the schedule settings so they can be saved or reused
        return {
            "starting_lr": self.starting_lr,
            "warmup_lr": self.warmup_lr,
            "final_lr": self.final_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "last_lr": self.get_lr(self.last_step),
            "last_step": self.last_step,
            "name": self.name,
        }