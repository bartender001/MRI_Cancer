import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any

class ResNetBlock(nn.Module):
    channels: int
    down_sample: bool = False

    @nn.compact
    def __call__(self, x):
        strides = (2, 1) if self.down_sample else (1, 1)
        residual = x

        x = nn.Conv(self.channels, (3, 3), strides=strides[0], padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)

        x = nn.Conv(self.channels, (3, 3), strides=strides[1], padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=False)(x)

        if self.down_sample:
            residual = nn.Conv(self.channels, (1, 1), strides=2, padding='SAME',
                               kernel_init=nn.initializers.kaiming_normal())(residual)
            residual = nn.BatchNorm(use_running_average=False)(residual)

        x += residual
        return nn.relu(x)

class ResNet18(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME',
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding='SAME')

        # Residual blocks
        for _ in range(2):
            x = ResNetBlock(64)(x)
        x = ResNetBlock(128, down_sample=True)(x)
        x = ResNetBlock(128)(x)
        x = ResNetBlock(256, down_sample=True)(x)
        x = ResNetBlock(256)(x)
        x = ResNetBlock(512, down_sample=True)(x)
        x = ResNetBlock(512)(x)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        x = nn.Dense(self.num_classes)(x)
        return nn.softmax(x)
